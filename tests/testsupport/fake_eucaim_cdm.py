"""Builds small synthetic EUCAIM CDM datasets for tests.

Tables are written from strings and DICOM series are generated with pydicom, so no
real data is committed. Most codes are placeholders, which is fine because the code
under test never runs CDM validation.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

CT_CODE = "EUCAIM:IMG1000042"  # "Computed tomography", in series_modality's code list
BODY_SITE_CODE = "EUCAIM:BP1000008"

PATIENT_IMG = "P001"  # patient with imaging
PATIENT_NO_IMG = "P002"  # patient without imaging

STUDY = "1.2.826.0.1.3680043.8.498.100"
SERIES_OK = "1.2.826.0.1.3680043.8.498.101"  # readable 3-slice series
SERIES_NO_DIR = "1.2.826.0.1.3680043.8.498.102"  # indexed, no folder on disk
SERIES_BAD_SIZE = "1.2.826.0.1.3680043.8.498.103"  # slices with different sizes
SERIES_UNREADABLE = "1.2.826.0.1.3680043.8.498.104"  # folder with non-DICOM files only

SERIES_OK_SHAPE = {"slices": 3, "rows": 4, "cols": 5}
SERIES_OK_RESCALE = {"slope": 2.0, "intercept": -5.0}

_TABLES: Dict[str, str] = {
    "patient": (
        "patient_id,patient_birth_sex,patient_managing_organization_id,"
        "patient_diagnostic_category\n"
        f"{PATIENT_IMG},EUCAIM:COM1000176,ORG-1,EUCAIM:COM1001087\n"
        f"{PATIENT_NO_IMG},EUCAIM:COM1000175,ORG-1,EUCAIM:COM1001087\n"
    ),
    "cancer_condition": (
        "cancer_condition_id,patient_id,procedure_id,cancer_condition_age_at_diagnosis,"
        "cancer_condition_age_unit,cancer_condition_code,cancer_condition_topography,"
        "cancer_condition_histology_morphology_behavior,cancer_condition_asserted_date,"
        "cancer_condition_type\n"
        f"CC1,{PATIENT_IMG},PR1,61,EUCAIM:CLIN1000001,EUCAIM:CLIN1000002,"
        "EUCAIM:CLIN1000003,EUCAIM:CLIN1000004,2020-01-15,EUCAIM:CLIN1000005\n"
        f"CC2,{PATIENT_NO_IMG},PR2,55.5,EUCAIM:CLIN1000001,ICD10:C50,"
        "EUCAIM:CLIN1000003,EUCAIM:CLIN1000004,2021-03-02,EUCAIM:CLIN1000005\n"
    ),
    "procedure": (
        "procedure_id,patient_id,procedure_code\n"
        f"PR1,{PATIENT_IMG},EUCAIM:CLIN1000010\n"
        f"PR2,{PATIENT_NO_IMG},EUCAIM:CLIN1000010\n"
        f"PR3,{PATIENT_IMG},EUCAIM:CLIN1000011\n"
    ),
    "episode": (
        "episode_id,patient_id,episode_type_code\n"
        f"EP1,{PATIENT_IMG},EUCAIM:CLIN1000020\n"
    ),
    "episode_event": (
        "episode_id,event_table_name,event_table_id\nEP1,procedure,PR1\n"
    ),
    # Only required columns: every optional column (e.g. treatment start offset) is
    # absent from the file.
    "treatment": (
        "treatment_id,patient_id,treatment_type\n"
        f"T1,{PATIENT_IMG},EUCAIM:CLIN1014933\n"
        f"T2,{PATIENT_IMG},EUCAIM:CLIN1014933\n"
    ),
    "surgical_procedure": (
        "treatment_id,patient_id,surgical_procedure_code,"
        "surgical_procedure_histopathology_finding\n"
        f"T1,{PATIENT_IMG},EUCAIM:CLIN1000030,EUCAIM:CLIN1000031\n"
    ),
    "radiotherapy": (
        "treatment_id,patient_id,radiotherapy_modality,"
        "radiotherapy_dose_delivered_to_volume\n"
        f"T2,{PATIENT_IMG},EUCAIM:CLIN1005273,EUCAIM:CLIN1000040\n"
    ),
    "image_study": (
        "study_uid,patient_id,procedure_id,study_acquisition_date\n"
        f"{STUDY},{PATIENT_IMG},PR3,2020-02-01\n"
    ),
    "image_series": (
        "series_uid,study_uid,series_modality,series_manufacturer_name,series_body_site\n"
        + "".join(
            f'{s},{STUDY},{CT_CODE},"Acme, Inc.",{BODY_SITE_CODE}\n'
            for s in (SERIES_OK, SERIES_NO_DIR, SERIES_BAD_SIZE, SERIES_UNREADABLE)
        )
    ),
}

TABLE_NAMES = tuple(_TABLES)


def write_dicom_series(
    series_dir: Path,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    sizes: Sequence[tuple],
    slope: float = 1.0,
    intercept: float = 0.0,
) -> None:
    """Writes one single-frame CT DICOM file per slice.

    Slice `k` is filled with the raw value `k + 1` and positioned at `z = 2.5 * k`.
    Files are named in reverse slice order so that the reader must sort them by
    position rather than by file name.

    Args:
        series_dir: Folder to write into (created if needed).
        patient_id: PatientID tag.
        study_uid: StudyInstanceUID tag.
        series_uid: SeriesInstanceUID tag.
        sizes: `(rows, cols)` for each slice.
        slope: RescaleSlope tag.
        intercept: RescaleIntercept tag.
    """
    series_dir.mkdir(parents=True, exist_ok=True)
    n = len(sizes)
    for k, (rows, cols) in enumerate(sizes):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0.0, 0.0, 2.5 * k]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [0.5, 0.7]
        ds.SliceThickness = 2.5
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = np.full((rows, cols), k + 1, dtype=np.int16).tobytes()

        ds.save_as(series_dir / f"slice_{n - k:03d}.dcm", enforce_file_format=True)


def build_cdm_dataset(
    root: Path,
    *,
    omit_tables: Iterable[str] = (),
    overrides: Optional[Dict[str, str]] = None,
) -> Path:
    """Writes a synthetic CDM dataset under `root`.

    Args:
        root: Dataset root folder (created if needed).
        omit_tables: Tables whose CSV file is not written.
        overrides: Replacement CSV content, keyed by table name.

    Returns:
        `root`.
    """
    overrides = overrides or {}
    clinical = root / "clinical_data"
    clinical.mkdir(parents=True, exist_ok=True)
    for name, content in _TABLES.items():
        if name in omit_tables:
            continue
        (clinical / f"{name}.csv").write_text(
            overrides.get(name, content), encoding="utf-8"
        )

    study_dir = root / "imaging_data" / PATIENT_IMG / STUDY
    shape = SERIES_OK_SHAPE
    write_dicom_series(
        study_dir / SERIES_OK,
        patient_id=PATIENT_IMG,
        study_uid=STUDY,
        series_uid=SERIES_OK,
        sizes=[(shape["rows"], shape["cols"])] * shape["slices"],
        **SERIES_OK_RESCALE,
    )
    write_dicom_series(
        study_dir / SERIES_BAD_SIZE,
        patient_id=PATIENT_IMG,
        study_uid=STUDY,
        series_uid=SERIES_BAD_SIZE,
        sizes=[(4, 5), (4, 6)],
    )
    unreadable = study_dir / SERIES_UNREADABLE
    unreadable.mkdir(parents=True)
    (unreadable / "notes.txt").write_text("not a DICOM file", encoding="utf-8")

    return root
