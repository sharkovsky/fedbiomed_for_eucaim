# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader for datasets in the EUCAIM Common Data Model (CDM).

Thin wrapper around `eucaim_cdm_reader.CdmReader`: it gives access to the normalized
`clinical_data/` tables, finds the series matching a `SeriesQuery` (modality and
acquisition parameters) and reads one DICOM series into a volume. CDM validation is
deliberately not run here (the data is assumed to be validated beforehand).
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd
from eucaim_cdm_reader import CdmReader, ParameterFilter, schema
from eucaim_cdm_reader.dicom_array import DicomVolume, series_to_volume
from eucaim_cdm_reader.query import filter_series
from eucaim_cdm_reader.schema import FieldSpec, bare_code

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from fedbiomed.common.logger import logger

# Parent of every logger of the library (`eucaim_cdm_reader.reader`, `...query`, ...)
_LIBRARY_LOGGER = logging.getLogger("eucaim_cdm_reader")


class _FedLoggerBridge(logging.Handler):
    """Hands the records of the library's loggers to the Fed-BioMed logger's handlers.

    `eucaim_cdm_reader` logs through the standard `logging` module, while the Fed-BioMed
    logger does not propagate, so without this bridge library messages only reach the
    interpreter's last-resort stderr handler and never the node's log outputs.

    Records are passed through unchanged: they keep their library logger name (shown by
    the Fed-BioMed formats) and carry neither `broadcast` nor `researcher_id`, so the gRPC
    handler never forwards them to a researcher. This matters because library messages
    contain local paths and patient identifiers. They only reach local outputs (console,
    application log file); the security log only accepts records flagged `is_security`.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Passes `record` to the Fed-BioMed handlers if the Fed-BioMed level allows it.

        Each Fed-BioMed handler then applies its own level.
        """
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def _bridge_library_logging() -> None:
    """Routes the library's log records to the Fed-BioMed logger. Idempotent."""
    if any(isinstance(h, _FedLoggerBridge) for h in _LIBRARY_LOGGER.handlers):
        return
    _LIBRARY_LOGGER.addHandler(_FedLoggerBridge())
    # Let every record reach the bridge: the Fed-BioMed level filters them instead of
    # the root logger's level (WARNING by default).
    _LIBRARY_LOGGER.setLevel(logging.DEBUG)
    # Records are output by the Fed-BioMed handlers only: do not also hand them to the
    # root logger's handlers (that would duplicate them when the root is configured).
    _LIBRARY_LOGGER.propagate = False


_bridge_library_logging()

# Normalized tables accessible through the reader, with whether the CDM makes them
# mandatory (`eucaim_cdm_reader.schema.MANDATORY_TABLES`, also used by its validation).
_TABLES: Dict[str, bool] = {
    name: name in schema.MANDATORY_TABLES
    for name in (
        "patient",
        "cancer_condition",
        "procedure",
        "episode",
        "episode_event",
        "image_study",
        "image_series",
        "treatment",
        "surgical_procedure",
        "radiotherapy",
    )
}


@dataclass(frozen=True)
class SeriesQuery:
    """Which image series provide one image modality of a sample.

    Mirrors the arguments of `eucaim_cdm_reader.CdmReader.find_series`: a series
    matches when its `series_modality` is `modality` (if given) and it satisfies every
    `ParameterFilter` on its acquisition parameters (rows of `image_modality.csv`).
    Codes match with or without their `EUCAIM:` prefix, in the query and in the data.
    In the query, a code may also be given as its label in the CDM schema, in any
    letter case (for example `"Magnetic resonance imaging"`).

    Example: T1-weighted MR series, i.e. MR with Tissue Contrast = T1 weighted::

        SeriesQuery(
            "EUCAIM:IMG1000038",
            parameters=ParameterFilter("IMG1016649", value_code="IMG1016652"),
        )

    Attributes:
        modality: `series_modality` code or label, or `None` to match any modality.
        parameters: Conditions on the acquisition parameters; a single
            `ParameterFilter` or an iterable of them (stored as a tuple).
        include_subclasses: Also match the EUCAIM ontology subclasses of `modality`
            and of each filter's `value_code`. Needs the ontology, i.e. the
            `eucaim-cdm-reader[ontology]` extra (rdflib).

    Codes are checked against the schema by `EucaimCdmReader.check_series_query`,
    not here.
    """

    modality: Optional[str] = None
    parameters: Union[ParameterFilter, Iterable[ParameterFilter]] = ()
    include_subclasses: bool = False

    def __post_init__(self) -> None:
        """Checks the types and stores `parameters` as a tuple.

        Raises:
            FedbiomedValueError: if an attribute has a wrong type, or if the query
                has neither a modality nor a parameter.
        """
        if self.modality is not None and not (
            isinstance(self.modality, str) and self.modality
        ):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: SeriesQuery `modality` must be a "
                f"non-empty str or None, got {self.modality!r}"
            )

        parameters = self.parameters
        if isinstance(parameters, ParameterFilter):
            parameters = (parameters,)
        elif isinstance(parameters, (str, dict)) or not isinstance(
            parameters, Iterable
        ):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: SeriesQuery `parameters` must be a "
                f"ParameterFilter or an iterable of them, got {parameters!r}"
            )
        parameters = tuple(parameters)
        if not all(isinstance(p, ParameterFilter) for p in parameters):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: SeriesQuery `parameters` must only "
                f"contain ParameterFilter objects, got {parameters!r}"
            )
        object.__setattr__(self, "parameters", parameters)

        if not isinstance(self.include_subclasses, bool):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: SeriesQuery `include_subclasses` must "
                f"be a bool, got {self.include_subclasses!r}"
            )
        if self.modality is None and not parameters:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: SeriesQuery needs a modality, a "
                "parameter, or both"
            )


def _empty_frame(string_columns: Iterable[str], **typed: str) -> pd.DataFrame:
    """Returns an empty DataFrame with `StringDtype` columns plus `typed` ones."""
    columns = {c: pd.Series(dtype="string") for c in string_columns}
    columns.update({c: pd.Series(dtype=dtype) for c, dtype in typed.items()})
    return pd.DataFrame(columns)


class EucaimCdmReader:
    """Reads the tables and DICOM series of one EUCAIM CDM dataset.

    The dataset root follows the EUCAIM layout::

        root/
        ├── clinical_data/{table}.csv
        └── imaging_data/{patient_id}/{study_uid}/{series_uid}/*.dcm

    Tables are loaded lazily on first access and cached by the underlying `CdmReader`.
    """

    MANDATORY_TABLES: Tuple[str, ...] = tuple(
        name for name, mandatory in _TABLES.items() if mandatory
    )
    OPTIONAL_TABLES: Tuple[str, ...] = tuple(
        name for name, mandatory in _TABLES.items() if not mandatory
    )

    def __init__(self, root: str | Path) -> None:
        """Constructs the reader.

        Args:
            root: Path to the dataset root folder (the folder that contains
                `clinical_data/`).

        Raises:
            FedbiomedError: if `root` is not an existing folder.
        """
        if not isinstance(root, (str, Path)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: EUCAIM CDM dataset root must be a str or "
                f"Path, got {type(root).__name__}"
            )
        self._root = Path(root)
        if not self._root.is_dir():
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: EUCAIM CDM dataset root is not a folder: "
                f"{self._root}"
            )
        self._cdm = CdmReader(self._root)

    def root(self) -> Path:
        """Returns the dataset root folder."""
        return self._root

    def table(self, name: str) -> Optional[pd.DataFrame]:
        """Returns one normalized table.

        Columns use the nullable pandas dtypes (`StringDtype`, `Float64`, `Int64`,
        `boolean`) and date columns are parsed to `datetime64[us]`. A value that cannot be
        converted to its column's type is read as missing (the library logs a warning).
        Columns that are not in the schema are dropped, and an optional column absent
        from the CSV is absent from the returned DataFrame.

        Args:
            name: Table name, i.e. the CSV file stem in `clinical_data/`
                (for example `"patient"`).

        Returns:
            A copy of the table, or `None` if `name` is an optional table whose CSV file
            is missing.

        Raises:
            FedbiomedError: if `name` is not a supported table, if a mandatory table is
                missing, or if the CSV cannot be loaded (for example a required column
                is missing).
        """
        if name not in _TABLES:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: unsupported EUCAIM CDM table '{name}'. "
                f"Supported tables: {sorted(_TABLES)}"
            )
        try:
            # `CdmReader` caches the table and returns a copy of it
            return self._cdm.table(name)
        except FileNotFoundError as e:
            if not _TABLES[name]:
                return None
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: mandatory EUCAIM CDM table '{name}' is "
                f"missing from {self._root}"
            ) from e
        except (ValueError, TypeError, OSError) as e:
            # ValueError covers a missing required column, and pandas parser and
            # decoding errors
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot load EUCAIM CDM table '{name}' "
                f"from {self._root}: {e}"
            ) from e

    @staticmethod
    def field_spec(table: str, column: str) -> FieldSpec:
        """Returns the schema specification of one column.

        Needs no data: the specification comes from the schema bundled with
        `eucaim_cdm_reader`.

        Args:
            table: Table name (for example `"treatment"`).
            column: Column name in that table (for example `"treatment_type"`).

        Returns:
            The column's `FieldSpec`.

        Raises:
            FedbiomedError: if the table is not supported or has no such column.
        """
        if table not in _TABLES:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: unsupported EUCAIM CDM table '{table}'. "
                f"Supported tables: {sorted(_TABLES)}"
            )
        try:
            return schema.TABLES[table].field(column)
        except KeyError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: EUCAIM CDM table '{table}' has no column "
                f"'{column}'"
            ) from e

    @staticmethod
    def check_series_query(query: SeriesQuery) -> None:
        """Checks the codes of a series query against the schema. Needs no data.

        The library's own selection runs on empty tables, so the rules are exactly
        those of `CdmReader.find_series`: parameter codes, value codes and units must
        be in the schema code lists, and with `include_subclasses` every code must be
        an EUCAIM ontology class. In addition, without `include_subclasses` the
        modality must be in the `series_modality` code list (the library only warns).

        Args:
            query: Query to check.

        Raises:
            FedbiomedValueError: if `query` is not a `SeriesQuery`, if a code is not
                allowed, or if `include_subclasses` is set but the ontology cannot be
                loaded (rdflib missing).
        """
        if not isinstance(query, SeriesQuery):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: expected a SeriesQuery, got "
                f"{type(query).__name__}"
            )

        if query.modality is not None and not query.include_subclasses:
            field = EucaimCdmReader.field_spec("image_series", "series_modality")
            try:
                code = bare_code(query.modality, field)
            except ValueError as e:  # a label shared by several codes
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB634.value}: invalid SeriesQuery {query!r}: {e}"
                ) from e
            if code not in field.codes:
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB634.value}: SeriesQuery modality "
                    f"{query.modality!r} is not a `series_modality` code of the "
                    "EUCAIM CDM schema"
                )

        try:
            filter_series(
                _empty_frame(["series_uid", "study_uid", "series_modality"]),
                _empty_frame(["study_uid", "patient_id"]),
                _empty_frame(
                    [
                        "series_uid",
                        "acquisition_parameter_code",
                        "acquisition_parameter_value_code",
                        "acquisition_parameter_value_unit",
                    ],
                    acquisition_parameter_value_number="Float64",
                ),
                modality=query.modality,
                parameters=query.parameters,
                include_subclasses=query.include_subclasses,
            )
        except (ValueError, TypeError, ImportError) as e:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB634.value}: invalid SeriesQuery {query!r}: {e}"
            ) from e

    def find_series(self, query: SeriesQuery) -> Set[str]:
        """Returns the `series_uid`s of the series matching `query`.

        The query is not checked against the schema first: call
        `check_series_query` for that.

        Args:
            query: Modality and acquisition parameters to match.

        Returns:
            The matching series UIDs. Empty if `query` has parameters but
            `image_modality.csv` is missing (a warning is logged).

        Raises:
            FedbiomedError: if `query` is invalid or the tables cannot be loaded.
        """
        if query.parameters:
            try:
                self._cdm.table("image_modality")  # loads and caches the table
            except FileNotFoundError:
                logger.warning(
                    "EUCAIM CDM: table 'image_modality' is absent from this node, no "
                    f"series can match the acquisition parameters of {query!r}"
                )
                return set()
            except (ValueError, TypeError, OSError) as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB634.value}: cannot load EUCAIM CDM table "
                    f"'image_modality' from {self._root}: {e}"
                ) from e

        try:
            matches = self._cdm.find_series(
                modality=query.modality,
                parameters=query.parameters,
                include_subclasses=query.include_subclasses,
            )
        except FileNotFoundError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot find series for {query!r}: index "
                f"table missing from {self._root}"
            ) from e
        except (ValueError, TypeError, ImportError, OSError) as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot find series for {query!r}: {e}"
            ) from e

        return {str(uid) for uid in matches["series_uid"].dropna()}

    def read_series(
        self,
        patient_id: str,
        study_uid: str,
        series_uid: str,
        *,
        ras: bool = True,
        xyz_order: bool = True,
        rescale: bool = True,
    ) -> DicomVolume:
        """Reads one DICOM series into a volume.

        The series is located through `image_study` and `image_series`, under
        `imaging_data/{patient_id}/{study_uid}/{series_uid}/`.

        Args:
            patient_id: Patient identifier.
            study_uid: Study Instance UID; the study must belong to the patient.
            series_uid: Series Instance UID; the series must belong to the study.
            ras: Convert from LPS to RAS coordinates.
            xyz_order: Return the array in (X, Y, Z) axis order instead of (Z, Y, X).
            rescale: Apply `RescaleSlope` / `RescaleIntercept`.

        Returns:
            The volume (`array`, `affine`, `spacing`).

        Raises:
            FedbiomedError: if the index tables cannot be loaded, if the series is not
                indexed for this patient and study, if no instance could be read, or if
                the instances cannot be assembled into a volume.
        """
        series_desc = (
            f"patient '{patient_id}', study '{study_uid}', series '{series_uid}'"
        )

        try:
            dicom = self._cdm.load_dicom(
                series_uid, study_uid=study_uid, patient_id=patient_id
            )
        except FileNotFoundError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot locate DICOM series for "
                f"{series_desc}: index table missing from {self._root}"
            ) from e
        except (ValueError, TypeError, OSError) as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot locate DICOM series for "
                f"{series_desc}: {e}"
            ) from e

        datasets = dicom.get(study_uid, {}).get(series_uid, [])
        if not datasets:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: no readable DICOM instance for "
                f"{series_desc} (series not indexed, folder missing, or files unreadable)"
            )

        # `series_to_volume` reports data-quality problems (multi-frame instances,
        # non-uniform slice spacing, missing geometry, ...) through `warnings`, which
        # bypasses `logging`. Capture them and log them on the library logger, so they
        # take the same local-only route as the library's other messages.
        # Note: `catch_warnings` swaps process-wide state, it is not thread-safe.
        caught: List[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                return series_to_volume(
                    datasets, ras=ras, xyz_order=xyz_order, rescale=rescale
                )
        except Exception as e:
            # Pixel decoding and stacking can fail in many ways (inconsistent slice
            # sizes, unsupported transfer syntax, corrupted pixel data, ...)
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot build a volume from DICOM series "
                f"for {series_desc}: {e}"
            ) from e
        finally:
            for w in caught:
                _LIBRARY_LOGGER.warning(
                    "%s for %s: %s", w.category.__name__, series_desc, w.message
                )
