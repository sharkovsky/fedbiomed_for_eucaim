# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller for datasets in the EUCAIM Common Data Model (CDM).

One sample per patient (patients taken from `patient.csv`). At construction the
controller indexes every patient, study and series. `select()` then narrows the index
to the samples a dataset needs: it picks, for each patient, the study that provides the
requested image modalities (timepoint selection) and loads the requested clinical tables.
An image modality is a named `SeriesQuery` (modality code and acquisition parameters,
matched by `eucaim_cdm_reader`'s `find_series`), or a bare `series_modality` code.
`get_sample()` then filters those tables down to the sample's rows.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import (
    EucaimCdmReader,
    SeriesQuery,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller

TIMEPOINT_POLICIES: Tuple[str, ...] = ("unique", "earliest", "latest")

# Tables whose rows a sample can carry, in schema order. Not supported: `episode_event`
# (pointers only), `image_modality`, segmentation tables and the mandatory views.
CLINICAL_TABLES: Tuple[str, ...] = (
    "patient",
    "cancer_condition",
    "procedure",
    "treatment",
    "surgical_procedure",
    "radiotherapy",
    "episode",
    "image_study",
    "image_series",
)

# Reasons for dropping a patient in `select()`
DROP_NO_IMAGING = "no_imaging"  # the patient has no study
DROP_MISSING_MODALITIES = "missing_modalities"  # no study has every requested modality
DROP_DUPLICATE_MODALITY = (
    "duplicate_modality"  # ... except with >1 series of a modality
)
DROP_SEVERAL_CANDIDATES = "several_candidates"  # policy "unique", >1 candidate study
DROP_NO_ORDERING = "no_ordering"  # policy "earliest"/"latest", candidates not orderable
DROP_TIE = "tie"  # policy "earliest"/"latest", several candidates share the extreme

_DATE_COL = "study_acquisition_date"
_OFFSET_COL = "study_offset_from_diagnosis"
_UNIT_COL = "study_offset_unit"


def _err(msg: str) -> FedbiomedError:
    """Builds a `FedbiomedError` with the EUCAIM CDM error code (FB634)."""
    return FedbiomedError(f"{ErrorNumbers.FB634.value}: {msg}")


class EucaimCdmController(Controller):
    """Controller for a EUCAIM CDM dataset.

    The dataset root follows the EUCAIM layout::

        root/
        ├── clinical_data/{table}.csv
        └── imaging_data/{patient_id}/{study_uid}/{series_uid}/*.dcm

    `patient`, `image_study` and `image_series` must be present. No CDM validation is
    run: the data is assumed to be validated beforehand. Rows that would make the index
    ambiguous (missing or duplicated identifiers, references to absent patients or
    studies) are ignored with a warning instead of failing the whole dataset.
    """

    _patients: List[str]  # usable patient IDs, in `patient.csv` order
    _studies: pd.DataFrame  # usable rows of `image_study`, all its columns
    _series: pd.DataFrame  # usable rows of `image_series`, all its columns

    def __init__(self, root: Union[str, Path]) -> None:
        """Constructs the controller.

        Args:
            root: Dataset root folder (the folder that contains `clinical_data/`).

        Raises:
            FedbiomedError: if `root` is not an existing folder, if a mandatory table
                (`patient`, `image_study`, `image_series`) is missing or cannot be
                loaded, or if no patient can be indexed.
        """
        super().__init__()
        self.root = root
        self._reader = EucaimCdmReader(self.root)

        self._build_index()
        # patient, study and series of each sample; None until `select()`
        self._samples: Optional[List[Dict[str, Any]]] = None
        # requested clinical tables, in schema order; None for a missing optional table
        self._clinical: Dict[str, Optional[pd.DataFrame]] = {}
        # Parameters stored at registration, used to rebuild the controller
        self._controller_kwargs = {"root": str(self.root)}

    # === Properties ===
    @property
    def patients(self) -> List[str]:
        """Indexed patient IDs, in `patient.csv` order."""
        return list(self._patients)

    @property
    def image_modalities(self) -> List[str]:
        """`series_modality` codes of the indexed series on this node, sorted."""
        return sorted(self._series["series_modality"].dropna().unique().tolist())

    @property
    def samples(self) -> Optional[List[Dict[str, Any]]]:
        """Selected samples, or `None` before `select()`.

        Each sample is `{"patient_id": str, "study_uid": str | None,
        "series": {modality: series_uid}}`.
        """
        if self._samples is None:
            return None
        return [{**s, "series": dict(s["series"])} for s in self._samples]

    @property
    def missing_tables(self) -> List[str]:
        """Requested optional tables absent from this node (last `select()`)."""
        return [name for name, df in self._clinical.items() if df is None]

    # === Index ===
    def _build_index(self) -> None:
        """Builds the patient/study/series index.

        Raises:
            FedbiomedError: if a mandatory table is missing or cannot be loaded, or if
                no patient can be indexed.
        """
        self._patients = self._index_patients(self._reader.table("patient"))
        self._studies = self._index_studies(
            self._reader.table("image_study"), set(self._patients)
        )
        self._series = self._index_series(
            self._reader.table("image_series"), set(self._studies["study_uid"])
        )

        logger.info(
            f"EUCAIM CDM dataset indexed: {len(self._patients)} patients, "
            f"{len(self._studies)} studies, {len(self._series)} series"
        )

    @staticmethod
    def _warn_ignored(table: str, count: int, reason: str) -> None:
        """Logs how many rows of `table` the index ignores, and why (if any)."""
        if count:
            logger.warning(f"EUCAIM CDM: {count} row(s) of '{table}' ignored: {reason}")

    @classmethod
    def _index_patients(cls, df: pd.DataFrame) -> List[str]:
        """Returns the usable patient IDs of `patient`, in file order.

        Rows with a missing ID, and every copy of a duplicated ID, are ignored.

        Raises:
            FedbiomedError: if no patient is left.
        """
        ids = df["patient_id"]
        cls._warn_ignored("patient", int(ids.isna().sum()), "missing patient_id")
        ids = ids.dropna()

        duplicated = ids.duplicated(keep=False)
        cls._warn_ignored(
            "patient", int(duplicated.sum()), "duplicated patient_id (all copies)"
        )
        patients = [str(p) for p in ids[~duplicated]]

        if not patients:
            raise _err("no patient can be indexed from the 'patient' table")
        return patients

    @classmethod
    def _index_studies(cls, df: pd.DataFrame, patients: set) -> pd.DataFrame:
        """Returns the usable rows of `image_study`.

        Ignores rows with a missing `study_uid` or `patient_id`, every copy of a
        duplicated `study_uid`, and studies of patients not in `patients`.
        """
        missing_id = df["study_uid"].isna() | df["patient_id"].isna()
        cls._warn_ignored(
            "image_study", int(missing_id.sum()), "missing study_uid or patient_id"
        )
        df = df[~missing_id]

        duplicated = df["study_uid"].duplicated(keep=False)
        cls._warn_ignored(
            "image_study", int(duplicated.sum()), "duplicated study_uid (all copies)"
        )
        df = df[~duplicated]

        orphan = ~df["patient_id"].isin(patients)
        cls._warn_ignored(
            "image_study", int(orphan.sum()), "patient_id not in usable patients"
        )
        df = df[~orphan].reset_index(drop=True)

        return df

    @classmethod
    def _index_series(cls, df: pd.DataFrame, studies: set) -> pd.DataFrame:
        """Returns the usable rows of `image_series`.

        Ignores rows with a missing `series_uid` or `study_uid`, every copy of a
        duplicated `series_uid`, and series of studies not in `studies`.
        """
        missing_id = df["series_uid"].isna() | df["study_uid"].isna()
        cls._warn_ignored(
            "image_series", int(missing_id.sum()), "missing series_uid or study_uid"
        )
        df = df[~missing_id]

        duplicated = df["series_uid"].duplicated(keep=False)
        cls._warn_ignored(
            "image_series", int(duplicated.sum()), "duplicated series_uid (all copies)"
        )
        df = df[~duplicated]

        orphan = ~df["study_uid"].isin(studies)
        cls._warn_ignored(
            "image_series", int(orphan.sum()), "study_uid not in usable studies"
        )
        df = df[~orphan].reset_index(drop=True)

        return df

    # === Selection ===
    def select(
        self,
        modalities: Union[Mapping[str, SeriesQuery], Iterable[str]] = (),
        timepoint_policy: str = "unique",
        tables: Iterable[str] = (),
    ) -> None:
        """Selects the samples and loads the requested clinical tables.

        Can be called again: each call recomputes the selection from the full index.
        On error, the previous selection is left unchanged.

        Args:
            modalities: Image modalities a sample must provide, either as
                `{name: SeriesQuery}` or as `series_modality` codes (for example
                `"EUCAIM:IMG1000042"`, shorthand for `{code: SeriesQuery(code)}`).
                A series provides a modality when it matches its query; one series may
                provide several modalities. All of a sample's image modalities come
                from one study. If empty, every patient is a sample.
            timepoint_policy: How to choose among a patient's candidate studies, i.e.
                studies with exactly one series of each requested modality:
                - `"unique"`: keep the patient only if there is a single candidate.
                - `"earliest"` / `"latest"`: order the candidates by
                  `study_acquisition_date`, or, if a candidate has no date, by
                  `study_offset_from_diagnosis` when all candidates share one
                  `study_offset_unit`. Drop the patient on a tie or if the
                  candidates cannot be ordered.
                Ignored if `modalities` is empty.
            tables: Clinical tables whose rows go into each sample (see
                `CLINICAL_TABLES`). `image_study` / `image_series` rows are those of
                the selected study / series when `modalities` is not empty, otherwise
                all of the patient's studies / series.

        Raises:
            FedbiomedError: if an argument is invalid, if a requested mandatory table
                is missing or cannot be loaded, or if no sample remains.
        """
        queries = self._normalize_modalities(modalities)
        modalities = tuple(queries)
        tables = self._normalize_tables(tables)
        if timepoint_policy not in TIMEPOINT_POLICIES:
            raise _err(
                f"`timepoint_policy` must be one of {list(TIMEPOINT_POLICIES)}, got "
                f"{timepoint_policy!r}"
            )

        samples, dropped = self._select_timepoints(queries, timepoint_policy)
        if not samples:
            raise _err(
                "no sample left after selection "
                f"(modalities={list(modalities)}, timepoint_policy="
                f"'{timepoint_policy}'); dropped patients by reason: {dict(dropped)}"
            )
        clinical = self._load_tables(tables)

        # Commit only once everything succeeded
        self._samples = samples
        self._clinical = clinical

        logger.info(
            f"EUCAIM CDM: {len(samples)} sample(s) selected out of "
            f"{len(self._patients)} patients (modalities={list(modalities)}, "
            f"timepoint_policy='{timepoint_policy}', tables={list(tables)})"
        )
        if dropped:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
            logger.warning(
                f"EUCAIM CDM: {sum(dropped.values())} patient(s) dropped: {reasons}"
            )
        for name in self.missing_tables:
            logger.warning(
                f"EUCAIM CDM: optional table '{name}' is absent from this node, "
                "samples carry no row for it"
            )

    @classmethod
    def _normalize_modalities(
        cls, modalities: Union[Mapping[str, SeriesQuery], Iterable[str]]
    ) -> Dict[str, SeriesQuery]:
        """Checks the requested image modalities and returns `{name: query}`, sorted
        by name. Bare codes become `{code: SeriesQuery(code)}`."""
        if isinstance(modalities, Mapping):
            names = cls._normalize_codes(list(modalities), "modalities")
            for name in names:
                EucaimCdmReader.check_series_query(modalities[name])
            return {name: modalities[name] for name in names}

        codes = cls._normalize_codes(modalities, "modalities")
        queries = {code: SeriesQuery(code) for code in codes}
        for query in queries.values():
            EucaimCdmReader.check_series_query(query)
        return queries

    @staticmethod
    def _normalize_codes(values: Iterable[str], arg: str) -> Tuple[str, ...]:
        """Checks that `values` (a str or an iterable of str) holds non-empty str, and
        returns them deduplicated and sorted. `arg` names the argument in errors."""
        if isinstance(values, str):
            values = [values]
        try:
            values = list(values)
        except TypeError as e:
            raise _err(f"`{arg}` must be an iterable of str") from e
        if not all(isinstance(v, str) and v for v in values):
            raise _err(f"`{arg}` must only contain non-empty str")
        return tuple(sorted(set(values)))

    @classmethod
    def _normalize_tables(cls, values: Iterable[str]) -> Tuple[str, ...]:
        """Checks that the requested tables are supported, and returns them
        deduplicated, in schema order (`CLINICAL_TABLES`)."""
        requested = set(cls._normalize_codes(values, "tables"))
        unsupported = requested.difference(CLINICAL_TABLES)
        if unsupported:
            raise _err(
                f"unsupported table(s) {sorted(unsupported)}; supported tables: "
                f"{list(CLINICAL_TABLES)}"
            )
        return tuple(t for t in CLINICAL_TABLES if t in requested)

    def _select_timepoints(
        self, queries: Dict[str, SeriesQuery], policy: str
    ) -> Tuple[List[Dict[str, Any]], Counter]:
        """Picks the study of each patient that provides every modality of `queries`.

        Returns the samples, in patient order, and the number of dropped patients
        by reason. Without modalities, every patient is a sample with no study.
        """
        dropped: Counter = Counter()
        if not queries:
            samples = [
                {"patient_id": p, "study_uid": None, "series": {}}
                for p in self._patients
            ]
            return samples, dropped
        modalities = tuple(queries)

        # study_uid -> {modality: [series_uid, ...]} for the requested modalities.
        # Only indexed series count: rows ignored by the index never match.
        found: Dict[str, Dict[str, List[str]]] = {}
        for name, query in queries.items():
            matching = self._series["series_uid"].isin(self._reader.find_series(query))
            for row in self._series[matching].itertuples(index=False):
                found.setdefault(row.study_uid, {}).setdefault(name, []).append(
                    row.series_uid
                )

        studies_of = {
            p: list(uids)
            for p, uids in self._studies.groupby("patient_id", sort=False)["study_uid"]
        }

        samples = []
        for patient in self._patients:
            patient_studies = studies_of.get(patient, [])
            if not patient_studies:
                dropped[DROP_NO_IMAGING] += 1
                continue

            candidates, ambiguous = [], False
            for study in patient_studies:
                codes = found.get(study, {})
                if all(c in codes for c in modalities):
                    if all(len(codes[c]) == 1 for c in modalities):
                        candidates.append(study)
                    else:
                        ambiguous = True
            if not candidates:
                dropped[
                    DROP_DUPLICATE_MODALITY if ambiguous else DROP_MISSING_MODALITIES
                ] += 1
                continue

            if len(candidates) == 1:
                chosen = candidates[0]
            elif policy == "unique":
                dropped[DROP_SEVERAL_CANDIDATES] += 1
                continue
            else:
                chosen, reason = self._order_candidates(candidates, policy)
                if chosen is None:
                    dropped[reason] += 1
                    continue

            samples.append(
                {
                    "patient_id": patient,
                    "study_uid": chosen,
                    "series": {c: found[chosen][c][0] for c in modalities},
                }
            )
        return samples, dropped

    def _order_candidates(
        self, candidates: List[str], policy: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Picks the earliest or latest candidate study, or gives the drop reason."""
        info = self._studies.set_index("study_uid").loc[candidates]
        dates = self._column(info, _DATE_COL)
        if dates.notna().all():
            keys = dates
        else:
            offsets, units = (
                self._column(info, _OFFSET_COL),
                self._column(info, _UNIT_COL),
            )
            if offsets.notna().all() and units.notna().all() and units.nunique() == 1:
                keys = offsets
            else:
                return None, DROP_NO_ORDERING

        target = keys.min() if policy == "earliest" else keys.max()
        best = keys.index[keys == target]
        if len(best) != 1:
            return None, DROP_TIE
        return best[0], None

    @staticmethod
    def _column(df: pd.DataFrame, name: str) -> pd.Series:
        """Returns column `name` of `df`, or all missing values if this optional column
        is absent from the CSV."""
        if name in df:
            return df[name]
        return pd.Series(pd.NA, index=df.index, dtype="object")

    def _load_tables(
        self, tables: Tuple[str, ...]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Loads the requested clinical tables (`None` for a missing optional table).

        `image_study` and `image_series` come from the index, so that rows ignored when
        indexing are not handed out.
        """
        loaded: Dict[str, Optional[pd.DataFrame]] = {}
        for name in tables:
            if name == "image_study":
                loaded[name] = self._studies
            elif name == "image_series":
                loaded[name] = self._series
            else:
                loaded[name] = self._reader.table(name)
        return loaded

    # === Samples ===
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Returns one sample, without transforms.

        Returns:
            `{"clinical": {table: rows}, "<modality>": DicomVolume, ...}`. `"clinical"` is
            present only if tables were selected; `rows` is a DataFrame of this sample's
            rows of that table (no column if the optional table is missing). One
            volume per selected image modality.

        Raises:
            FedbiomedError: if `select()` was not called, or if a series cannot be read.
            IndexError: if `index` is out of range.
        """
        if self._samples is None:
            raise _err("`select()` must be called before `get_sample()`")
        sample = self._samples[index]

        data: Dict[str, Any] = {}
        if self._clinical:
            data["clinical"] = {
                name: self._sample_rows(name, df, sample)
                for name, df in self._clinical.items()
            }

        for modality, series_uid in sample["series"].items():
            data[modality] = self._reader.read_series(
                sample["patient_id"], sample["study_uid"], series_uid
            )
        return data

    def _sample_rows(
        self, name: str, df: Optional[pd.DataFrame], sample: Dict[str, Any]
    ) -> pd.DataFrame:
        """Returns the rows of table `name` that belong to `sample`, as a new DataFrame.

        `image_study` / `image_series` rows are those of the selected study / series
        when image modalities were selected, otherwise all of the patient's.
        """
        if df is None:  # missing optional table
            return pd.DataFrame()

        selected_study = sample["study_uid"]  # None if no image modality selected
        if name == "image_series":
            if selected_study is not None:
                mask = df["series_uid"].isin(list(sample["series"].values()))
            else:
                is_patient = self._studies["patient_id"] == sample["patient_id"]
                patient_studies = self._studies.loc[is_patient, "study_uid"]
                mask = df["study_uid"].isin(patient_studies)
        elif name == "image_study" and selected_study is not None:
            mask = df["study_uid"] == selected_study
        else:
            mask = df["patient_id"] == sample["patient_id"]

        # A missing identifier gives NA in the mask: never a match
        return df[mask.fillna(False)].reset_index(drop=True)

    def __len__(self) -> int:
        """Number of selected samples, or of indexed patients before `select()`."""
        if self._samples is not None:
            return len(self._samples)
        return len(self._patients)

    def shape(self) -> Dict[str, Any]:
        """Describes the indexed data without reading any DICOM file.

        Returns:
            `{"patients": int, "series_per_modality": {code: int}}`.
        """
        counts = self._series["series_modality"].dropna().value_counts()
        return {
            "patients": len(self._patients),
            "series_per_modality": {
                str(k): int(counts[k]) for k in sorted(counts.index)
            },
        }

    def get_types(self) -> Dict[str, str]:
        """Types of the values a sample can hold, without reading any DICOM file."""
        return {
            "clinical": "dict",
            **{code: "DicomVolume" for code in self.image_modalities},
        }

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        """Returns the dataset type handled by this controller."""
        return DatasetTypes.EUCAIM_CDM
