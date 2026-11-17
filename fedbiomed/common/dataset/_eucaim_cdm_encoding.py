# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Encodes the clinical rows of a EUCAIM CDM sample into a 1-D float vector.

The layout of the vector depends only on the schema and on the researcher's arguments,
never on local data, so it is identical on every node:

- a coded column (with a schema code list) gives one 0/1 entry per vocabulary code: one-hot
  for a one-row table, multi-hot (1 if any row has the code) for a many-row table;
- a numeric or boolean column gives one entry: the value for a one-row table, or the
  `min` / `max` / `mean` of the rows for a many-row table (reducer required);
- dates, identifiers and free text are rejected.

Missing values give `NaN` (numbers) or zeros (codes).
"""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from eucaim_cdm_reader.schema import BOOLEAN, CODE, DATE, INTEGER, NUMBER

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._eucaim_cdm_controller import CLINICAL_TABLES
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import EucaimCdmReader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from fedbiomed.common.logger import logger

EUCAIM_PREFIX = "EUCAIM:"
REDUCERS = ("min", "max", "mean")

# Numeric columns and the unit column their value is expressed in (same table)
UNIT_PAIRS: Dict[str, str] = {
    "cancer_condition.cancer_condition_age_at_diagnosis": (
        "cancer_condition.cancer_condition_age_unit"
    ),
    "cancer_condition.cancer_condition_offset_from_primary": (
        "cancer_condition.cancer_condition_offset_unit"
    ),
    "procedure.procedure_offset_from_diagnosis": "procedure.procedure_offset_unit",
    "treatment.start_offset_from_diagnosis": "treatment.start_offset_unit",
    "treatment.end_offset_from_diagnosis": "treatment.end_offset_unit",
    "radiotherapy.radiotherapy_total_dose_delivered": (
        "radiotherapy.radiotherapy_dose_unit"
    ),
    "episode.start_offset_from_diagnosis": "episode.start_offset_unit",
    "episode.end_offset_from_diagnosis": "episode.end_offset_unit",
    "episode.episode_effective": "episode.episode_effective_unit",
    "image_study.study_offset_from_diagnosis": "image_study.study_offset_unit",
}


def _err(msg: str) -> FedbiomedValueError:
    """Builds a `FedbiomedValueError` with the EUCAIM CDM error code (FB634)."""
    return FedbiomedValueError(f"{ErrorNumbers.FB634.value}: {msg}")


def is_one_row_table(table: str, with_images: bool) -> bool:
    """Tells whether `table` gives exactly one row per sample (D1).

    `patient` always does; `image_study` does when image modalities are requested
    (the selected study). Every other table can give several rows.
    """
    return table == "patient" or (table == "image_study" and with_images)


def _bare_code(value: str) -> Optional[str]:
    """Returns `value` without its `EUCAIM:` prefix, or `None` for an external code.

    A value with another prefix (`ICD10:...`) never matches an EUCAIM code.
    """
    if value.startswith(EUCAIM_PREFIX):
        return value[len(EUCAIM_PREFIX) :]
    return None if ":" in value else value


class _ColumnEncoder:
    """Encodes one `table.column` of a sample's rows. Subclasses set the layout."""

    def __init__(self, table: str, column: str, one_row: bool):
        self.table = table
        self.column = column
        self.name = f"{table}.{column}"
        self.one_row = one_row
        self._warned_missing = False

    def feature_names(self) -> List[str]:
        """Names of the entries this column contributes to the vector."""
        raise NotImplementedError

    def encode(self, rows: Optional[pd.DataFrame]) -> np.ndarray:
        """Encodes this column of `rows` (the sample's rows of the table)."""
        if rows is None or self.column not in rows.columns:
            if not self._warned_missing:
                logger.warning(
                    f"EUCAIM CDM: column '{self.name}' is absent on this node, "
                    "encoded as if the sample had no row"
                )
                self._warned_missing = True
            rows = pd.DataFrame({self.column: pd.Series([], dtype="object")})
        if self.one_row and len(rows) > 1:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: table '{self.table}' should give one row "
                f"per sample, got {len(rows)}"
            )
        return self._encode(rows)

    def _encode(self, rows: pd.DataFrame) -> np.ndarray:
        """Encodes the column of `rows`, which is present."""
        raise NotImplementedError


class _CodedEncoder(_ColumnEncoder):
    """One-hot / multi-hot over a vocabulary of codes (bare, without prefix)."""

    def __init__(self, table: str, column: str, one_row: bool, vocabulary: List[str]):
        super().__init__(table, column, one_row)
        self.vocabulary = vocabulary
        self._position = {code: i for i, code in enumerate(vocabulary)}

    def feature_names(self) -> List[str]:
        """One name per code: `table.column=EUCAIM:CODE`."""
        return [f"{self.name}={EUCAIM_PREFIX}{code}" for code in self.vocabulary]

    def _encode(self, rows: pd.DataFrame) -> np.ndarray:
        """1 for each vocabulary code held by any row, 0 otherwise."""
        out = np.zeros(len(self.vocabulary), dtype=np.float64)
        for value in rows[self.column].dropna():
            position = self._position.get(_bare_code(str(value)))
            if position is not None:
                out[position] = 1.0
        return out


class _NumericEncoder(_ColumnEncoder):
    """Numeric or boolean value, reduced over the rows of a many-row table."""

    def __init__(
        self,
        table: str,
        column: str,
        one_row: bool,
        reducer: Optional[str],
        unit_column: Optional[str],
    ):
        super().__init__(table, column, one_row)
        self.reducer = reducer
        self.unit_column = unit_column
        self._warned_units = False

    def feature_names(self) -> List[str]:
        """`table.column`, or `table.column:reducer` for a many-row table."""
        return [self.name if self.reducer is None else f"{self.name}:{self.reducer}"]

    @staticmethod
    def _values(series: pd.Series) -> np.ndarray:
        """The column as floats (`True` / `False` give 1 / 0), `NaN` for missing values.

        The reader has already typed the column (`Int64`, `Float64` or `boolean`) and
        read unconvertible values as missing.
        """
        return series.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)

    def _encode(self, rows: pd.DataFrame) -> np.ndarray:
        """The value (one-row table) or the reduced values (many-row table)."""
        values = self._values(rows[self.column])
        present = ~np.isnan(values)
        if not present.any():
            return np.array([np.nan])
        if self.reducer is None:  # one-row table
            return values[:1]

        if self.unit_column is not None and present.sum() > 1:
            units = (
                rows[self.unit_column]
                if self.unit_column in rows.columns
                else pd.Series(pd.NA, index=rows.index, dtype="object")
            )[present]
            if units.isna().any() or units.nunique() > 1:
                if not self._warned_units:
                    logger.warning(
                        f"EUCAIM CDM: rows of '{self.name}' use different or unknown "
                        f"units ('{self.unit_column}'), reduced to NaN"
                    )
                    self._warned_units = True
                return np.array([np.nan])

        reduce = {"min": np.min, "max": np.max, "mean": np.mean}[self.reducer]
        return np.array([float(reduce(values[present]))])


class EncodingPlan:
    """Encodes the clinical rows of a sample for a fixed list of columns.

    Built from the schema and the researcher's arguments only: no data is needed, and
    the layout (`feature_names()`) is the same on every node.
    """

    def __init__(
        self,
        columns: Iterable[str],
        vocabularies: Optional[Mapping[str, Iterable[str]]] = None,
        reducers: Optional[Mapping[str, str]] = None,
        with_images: bool = False,
    ):
        """Builds the plan.

        Args:
            columns: `table.column` names, in output order.
            vocabularies: Codes to encode for some coded columns (default: the
                column's full code list, sorted). Entries for other columns are
                ignored.
            reducers: `min` / `max` / `mean` for each numeric column of a many-row
                table. Entries for other columns are ignored.
            with_images: Whether image modalities are requested, which makes
                `image_study` a one-row table.

        Raises:
            FedbiomedValueError: if a column cannot be encoded, a vocabulary is not a
                subset of the code list, or a reducer is missing, invalid, or given
                for a column that does not take one.
        """
        vocabularies = dict(vocabularies or {})
        reducers = dict(reducers or {})
        names = parse_columns(columns, "columns")

        self._encoders: List[_ColumnEncoder] = []
        for name in names:
            table, column = name.split(".", 1)
            spec = EucaimCdmReader.field_spec(table, column)
            one_row = is_one_row_table(table, with_images)

            if spec.kind == CODE:
                if name in reducers:
                    raise _err(f"reducer given for coded column '{name}'")
                vocabulary = (
                    normalize_vocabulary(name, vocabularies[name], spec.codes)
                    if name in vocabularies
                    else sorted(spec.codes)
                )
                self._encoders.append(_CodedEncoder(table, column, one_row, vocabulary))
                continue

            if name in vocabularies:
                raise _err(f"vocabulary given for column '{name}', which is not coded")

            if spec.kind not in (BOOLEAN, INTEGER, NUMBER):
                kind = "a date" if spec.kind == DATE else "an identifier or free text"
                raise _err(f"column '{name}' is {kind} and cannot be encoded")

            reducer = reducers.get(name)
            if one_row and reducer is not None:
                raise _err(
                    f"reducer given for '{name}', but '{table}' gives one row per sample"
                )
            if not one_row and reducer is None:
                raise _err(
                    f"'{table}' can give several rows per sample: a reducer "
                    f"({', '.join(REDUCERS)}) is required for '{name}'"
                )
            if reducer is not None and reducer not in REDUCERS:
                raise _err(
                    f"reducer for '{name}' must be one of {list(REDUCERS)}, got "
                    f"{reducer!r}"
                )
            unit = UNIT_PAIRS.get(name)
            self._encoders.append(
                _NumericEncoder(
                    table,
                    column,
                    one_row,
                    reducer,
                    None if unit is None else unit.split(".", 1)[1],
                )
            )

    @property
    def tables(self) -> Tuple[str, ...]:
        """Tables the plan reads, in first-use order."""
        return tuple(dict.fromkeys(e.table for e in self._encoders))

    def feature_names(self) -> List[str]:
        """Names of the vector entries, in order."""
        return [n for e in self._encoders for n in e.feature_names()]

    def encode(self, rows: Mapping[str, pd.DataFrame]) -> np.ndarray:
        """Encodes a sample's rows (`{table: rows}`) into a 1-D float64 vector."""
        return np.concatenate(
            [e.encode(rows.get(e.table)) for e in self._encoders]
        ).astype(np.float64)


def parse_columns(columns: Iterable[str], arg: str) -> List[str]:
    """Checks `table.column` names (supported table, no duplicates) and lists them.

    Raises:
        FedbiomedValueError: if a name is malformed, its table unsupported, or a name
            is repeated.
    """
    if isinstance(columns, str):
        columns = [columns]
    try:
        names = list(columns)
    except TypeError as e:
        raise _err(f"`{arg}` must be an iterable of 'table.column' str") from e
    if not names:
        raise _err(f"`{arg}` cannot be empty")

    for name in names:
        if not isinstance(name, str) or "." not in name:
            raise _err(f"`{arg}` entries must be 'table.column' str, got {name!r}")
        table = name.split(".", 1)[0]
        if table not in CLINICAL_TABLES:
            raise _err(
                f"unsupported table '{table}' in `{arg}`; supported tables: "
                f"{list(CLINICAL_TABLES)}"
            )
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise _err(f"duplicated column(s) in `{arg}`: {duplicated}")
    return names


def normalize_vocabulary(
    name: str, codes: Iterable[str], code_list: Iterable[str]
) -> List[str]:
    """Checks a researcher vocabulary and returns its bare codes, deduplicated.

    Codes are accepted with or without the `EUCAIM:` prefix and must be in the
    column's code list. The order given is kept.

    Raises:
        FedbiomedValueError: if the vocabulary is empty or holds an invalid code.
    """
    if isinstance(codes, str):
        codes = [codes]
    allowed = set(code_list)
    vocabulary: List[str] = []
    for code in codes:
        bare = _bare_code(code) if isinstance(code, str) else None
        if bare not in allowed:
            raise _err(f"code {code!r} is not in the code list of '{name}'")
        if bare not in vocabulary:
            vocabulary.append(bare)
    if not vocabulary:
        raise _err(f"vocabulary for '{name}' is empty")
    return vocabulary
