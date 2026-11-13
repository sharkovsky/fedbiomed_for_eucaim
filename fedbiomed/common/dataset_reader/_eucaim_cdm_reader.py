# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader for datasets in the EUCAIM Common Data Model (CDM).

Thin wrapper around `eucaim_cdm_reader.CdmReader`: it gives access to the normalized
`clinical_data/` tables and reads one DICOM series into a volume. CDM validation is
deliberately not run here (the data is assumed to be validated beforehand).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from eucaim_cdm_reader import CdmReader, DicomVolume, schema, series_to_volume
from eucaim_cdm_reader.schema import FieldSpec

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

# Parent of every logger of the library (`eucaim_cdm_reader.reader`, `..._utils`, ...)
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
        # The Fed-BioMed level decides, then each Fed-BioMed handler applies its own.
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

# Normalized tables accessible through the reader, with their schema field list and
# whether the CDM makes them mandatory. The split mirrors `CdmReader.validate()`.
_TABLES: Dict[str, Tuple[List[FieldSpec], bool]] = {
    "patient": (schema.PATIENT_FIELDS, True),
    "cancer_condition": (schema.CANCER_CONDITION_FIELDS, True),
    "procedure": (schema.PROCEDURE_FIELDS, True),
    "episode": (schema.EPISODE_FIELDS, True),
    "episode_event": (schema.EPISODE_EVENT_FIELDS, True),
    "image_study": (schema.IMAGE_STUDY_FIELDS, True),
    "image_series": (schema.IMAGE_SERIES_FIELDS, True),
    "treatment": (schema.TREATMENT_FIELDS, False),
    "surgical_procedure": (schema.SURGICAL_PROCEDURE_FIELDS, False),
    "radiotherapy": (schema.RADIOTHERAPY_FIELDS, False),
}


class EucaimCdmReader:
    """Reads the tables and DICOM series of one EUCAIM CDM dataset.

    The dataset root follows the EUCAIM layout::

        root/
        ├── clinical_data/{table}.csv
        └── imaging_data/{patient_id}/{study_uid}/{series_uid}/*.dcm

    Tables are loaded lazily on first access and cached by the underlying `CdmReader`.
    """

    MANDATORY_TABLES: Tuple[str, ...] = tuple(
        name for name, (_, mandatory) in _TABLES.items() if mandatory
    )
    OPTIONAL_TABLES: Tuple[str, ...] = tuple(
        name for name, (_, mandatory) in _TABLES.items() if not mandatory
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
        # Uses the rule exceptions bundled with the library (its default).
        self._cdm = CdmReader(self._root)

    def root(self) -> Path:
        """Returns the dataset root folder."""
        return self._root

    def table(self, name: str) -> Optional[pd.DataFrame]:
        """Returns one normalized table.

        Columns use the nullable pandas dtypes (`StringDtype`, `Float64`, `Int64`) and
        date columns are parsed to timestamps. An optional column absent from the CSV is
        absent from the returned DataFrame.

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
        _, mandatory = _TABLES[name]

        try:
            df = getattr(self._cdm, f"{name}_data")
        except FileNotFoundError as e:
            if not mandatory:
                return None
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: mandatory EUCAIM CDM table '{name}' is "
                f"missing from {self._root}"
            ) from e
        except (ValueError, TypeError, OSError) as e:
            # ValueError also covers pandas parser errors and decoding errors
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: cannot load EUCAIM CDM table '{name}' "
                f"from {self._root}: {e}"
            ) from e

        # The underlying reader caches the DataFrame: never hand out the cached object.
        return df.copy()

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
        fields, _ = _TABLES[table]
        for spec in fields:
            if spec.name == column:
                return spec
        raise FedbiomedError(
            f"{ErrorNumbers.FB634.value}: EUCAIM CDM table '{table}' has no column "
            f"'{column}'"
        )

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
            dicom = self._cdm.get_dicom_data(
                patient_id=patient_id, study_uid=study_uid, series_uid=series_uid
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
