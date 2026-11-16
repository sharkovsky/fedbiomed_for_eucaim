"""Builds small synthetic EUCAIM CDM datasets for tests.

Tables are written from strings and DICOM series are generated with pydicom, so no
real data is committed. Some codes are placeholders, which is fine because the code
under test never runs CDM validation.

Imaging layout (`date` is `study_acquisition_date`, `offset` is
`study_offset_from_diagnosis` with its unit; `DICOM` marks series written on disk):

    patient  study  date        offset    series (modality)
    P001     S1     2020-02-01  30 day    SE1_CT (CT, DICOM 3 slices), SE1_MR (MR, DICOM)
    P002     -                            (no imaging)
    P003     S3A    2020-01-01  10 day    SE3A_CT (CT, DICOM)
             S3B    -           40 day    SE3B_CT (CT, DICOM)
    P004     S4     2020-03-01  -         SE4_CT1 + SE4_CT2 (CT twice), SE4_MR (MR, DICOM)
    P005     S5A    -           5 day     SE5A_CT (CT)
             S5B    -           7 month   SE5B_CT (CT)
    P006     S6A    2020-05-05  -         SE6A_CT (CT)
             S6B    2020-05-05  -         SE6B_CT (CT)
    P007     S7     2020-06-01  -         SE7 (PCCT, indexed but no folder on disk)
    P008     S8     2020-07-01  -         SE8_BAD (PET, slices of different sizes),
                                          SE8_UNREADABLE (SPECT, non-DICOM file only)
    P999     S_ORPHAN                     (study of a patient absent from `patient`)
    -        -                            SE_ORPHAN (series of a study absent from
                                          `image_study`)

Expected timepoint selection:
    {CT}  unique:   P001/S1
    {CT}  earliest: P001/S1, P003/S3A (offset fallback: S3B is undated)
    {CT}  latest:   P001/S1, P003/S3B
    {MR}  any:      P001/S1, P004/S4
    {CT, MR} any:   P001/S1 (S4 has CT twice)

Acquisition parameters (`image_modality.csv`, written separately from the other
tables; `-` marks a code stored without its `EUCAIM:` prefix):

    series     Tissue Contrast   Slice Thickness
    SE1_MR     T1 weighted       0.25 cm
    SE4_MR     T2 weighted       5 mm
    SE1_CT     -                 2.5 mm (codes without prefix)
    SE_ORPHAN  T1 weighted       -

`add_mr_sequences_patient()` adds a patient P009 whose study S9 holds two MR series,
SE9_T1 (T1 weighted) and SE9_T2 (T2 weighted), both on disk: as plain MR the study is
not a candidate (MR twice), but as T1 + T2 it is.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# series_modality codes, all in the schema code list
CT_CODE = "EUCAIM:IMG1000042"  # Computed tomography
MR_CODE = "EUCAIM:IMG1000038"  # Magnetic resonance imaging
PET_CODE = "EUCAIM:IMG1000062"  # Positron emission tomography
SPECT_CODE = "EUCAIM:IMG1000061"  # Single photon emission computed tomography
PCCT_CODE = "EUCAIM:IMG1000085"  # Photon Counting Computed Tomography
BODY_SITE_CODE = "EUCAIM:BP1000008"

DAY = "EUCAIM:COM1000153"
MONTH = "EUCAIM:COM1000154"
MALE = "EUCAIM:COM1001366"  # patient_birth_sex
FEMALE = "EUCAIM:COM1001370"
CHEMO = "EUCAIM:CLIN1024528"  # treatment_type: Chemotherapy

# image_modality codes
TISSUE_CONTRAST = "EUCAIM:IMG1016649"  # acquisition_parameter_code
T1 = "EUCAIM:IMG1016652"  # Tissue Contrast value: T1 weighted
T2 = "EUCAIM:IMG1016647"  # Tissue Contrast value: T2 weighted
SLICE_THICKNESS = "EUCAIM:IMG1016306"  # acquisition_parameter_code
MM = "EUCAIM:COM1000152"  # millimeter
CM = "EUCAIM:COM1001954"  # centimeter
BIO = "EUCAIM:CLIN1014763"  # treatment_type: Biological Therapy


def _uid(n: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.{n}"


PATIENTS = ("P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008")

S1, S3A, S3B, S4 = _uid(100), _uid(300), _uid(310), _uid(400)
S5A, S5B, S6A, S6B = _uid(500), _uid(510), _uid(600), _uid(610)
S7, S8, S_ORPHAN = _uid(700), _uid(800), _uid(990)

SE1_CT, SE1_MR = _uid(101), _uid(102)
SE3A_CT, SE3B_CT = _uid(301), _uid(311)
SE4_CT1, SE4_CT2, SE4_MR = _uid(401), _uid(402), _uid(403)
SE5A_CT, SE5B_CT = _uid(501), _uid(511)
SE6A_CT, SE6B_CT = _uid(601), _uid(611)
SE7 = _uid(701)
SE8_BAD, SE8_UNREADABLE = _uid(801), _uid(802)
SE_ORPHAN = _uid(991)

# Added by `add_mr_sequences_patient()`
PATIENT_MR_SEQ, S9, SE9_T1, SE9_T2 = "P009", _uid(900), _uid(901), _uid(902)

# Names used by the reader tests
PATIENT_IMG, PATIENT_NO_IMG = "P001", "P002"
STUDY, SERIES_OK = S1, SE1_CT
PATIENT_NO_DIR, STUDY_NO_DIR, SERIES_NO_DIR = "P007", S7, SE7
PATIENT_BROKEN, STUDY_BROKEN = "P008", S8
SERIES_BAD_SIZE, SERIES_UNREADABLE = SE8_BAD, SE8_UNREADABLE

SERIES_OK_SHAPE = {"slices": 3, "rows": 4, "cols": 5}
SERIES_OK_RESCALE = {"slope": 2.0, "intercept": -5.0}

# (patient, study, date, offset, unit)
_STUDIES = (
    ("P001", S1, "2020-02-01", "30", DAY),
    ("P003", S3A, "2020-01-01", "10", DAY),
    ("P003", S3B, "", "40", DAY),
    ("P004", S4, "2020-03-01", "", ""),
    ("P005", S5A, "", "5", DAY),
    ("P005", S5B, "", "7", MONTH),
    ("P006", S6A, "2020-05-05", "", ""),
    ("P006", S6B, "2020-05-05", "", ""),
    ("P007", S7, "2020-06-01", "", ""),
    ("P008", S8, "2020-07-01", "", ""),
    ("P999", S_ORPHAN, "2020-08-01", "", ""),
)

# (study, series, modality)
_SERIES = (
    (S1, SE1_CT, CT_CODE),
    (S1, SE1_MR, MR_CODE),
    (S3A, SE3A_CT, CT_CODE),
    (S3B, SE3B_CT, CT_CODE),
    (S4, SE4_CT1, CT_CODE),
    (S4, SE4_CT2, CT_CODE),
    (S4, SE4_MR, MR_CODE),
    (S5A, SE5A_CT, CT_CODE),
    (S5B, SE5B_CT, CT_CODE),
    (S6A, SE6A_CT, CT_CODE),
    (S6B, SE6B_CT, CT_CODE),
    (S7, SE7, PCCT_CODE),
    (S8, SE8_BAD, PET_CODE),
    (S8, SE8_UNREADABLE, SPECT_CODE),
    (_uid(999), SE_ORPHAN, CT_CODE),
)

# (patient, study, series, number of slices) for readable series written on disk
_READABLE_SERIES = (
    ("P001", S1, SE1_MR, 2),
    ("P003", S3A, SE3A_CT, 2),
    ("P003", S3B, SE3B_CT, 2),
    ("P004", S4, SE4_MR, 2),
)

_TABLES: Dict[str, str] = {
    "patient": (
        "patient_id,patient_birth_sex,patient_managing_organization_id,"
        "patient_diagnostic_category\n"
        # Odd patients (P001, P003, ...) are male, even ones female
        + "".join(
            f"{p},{(MALE, FEMALE)[i % 2]},ORG-1,EUCAIM:COM1001087\n"
            for i, p in enumerate(PATIENTS)
        )
    ),
    "cancer_condition": (
        "cancer_condition_id,patient_id,procedure_id,cancer_condition_age_at_diagnosis,"
        "cancer_condition_age_unit,cancer_condition_code,cancer_condition_topography,"
        "cancer_condition_histology_morphology_behavior,cancer_condition_asserted_date,"
        "cancer_condition_type\n"
        "CC1,P001,PR1,61,EUCAIM:CLIN1000001,EUCAIM:CLIN1000002,"
        "EUCAIM:CLIN1000003,EUCAIM:CLIN1000004,2020-01-15,EUCAIM:CLIN1000005\n"
        "CC2,P002,PR2,55.5,EUCAIM:CLIN1000001,ICD10:C50,"
        "EUCAIM:CLIN1000003,EUCAIM:CLIN1000004,2021-03-02,EUCAIM:CLIN1000005\n"
        "CC3,P003,PR3,70,EUCAIM:CLIN1000001,EUCAIM:CLIN1000002,"
        "EUCAIM:CLIN1000003,EUCAIM:CLIN1000004,2019-12-20,EUCAIM:CLIN1000005\n"
    ),
    # Only required columns: every optional column (e.g. procedure_offset_unit) is
    # absent from the file.
    "procedure": (
        "procedure_id,patient_id,procedure_code\n"
        "PR1,P001,EUCAIM:CLIN1000010\n"
        "PR2,P002,EUCAIM:CLIN1000010\n"
        "PR3,P003,EUCAIM:CLIN1000010\n"
        "PR_IMG,P001,EUCAIM:CLIN1000011\n"
    ),
    "episode": "episode_id,patient_id,episode_type_code\nEP1,P001,EUCAIM:CLIN1000020\n",
    "episode_event": (
        "episode_id,event_table_name,event_table_id\nEP1,procedure,PR1\n"
    ),
    # P003 has treatments whose offsets use different units
    "treatment": (
        "treatment_id,patient_id,treatment_type,start_offset_from_diagnosis,"
        "start_offset_unit\n"
        f"T1,P001,{CHEMO},10,{DAY}\n"
        f"T2,P001,{CHEMO},20,{DAY}\n"
        f"T3,P003,{CHEMO},2,{MONTH}\n"
        f"T4,P003,{BIO},30,{DAY}\n"
    ),
    "surgical_procedure": (
        "treatment_id,patient_id,surgical_procedure_code,"
        "surgical_procedure_histopathology_finding\n"
        "T1,P001,EUCAIM:CLIN1000030,EUCAIM:CLIN1000031\n"
    ),
    "radiotherapy": (
        "treatment_id,patient_id,radiotherapy_modality,"
        "radiotherapy_dose_delivered_to_volume\n"
        "T2,P001,EUCAIM:CLIN1005273,EUCAIM:CLIN1000040\n"
    ),
    "image_study": (
        "study_uid,patient_id,procedure_id,study_acquisition_date,"
        "study_offset_from_diagnosis,study_offset_unit\n"
        + "".join(
            f"{study},{patient},PR_IMG,{date},{offset},{unit}\n"
            for patient, study, date, offset, unit in _STUDIES
        )
    ),
    "image_series": (
        "series_uid,study_uid,series_modality,series_manufacturer_name,series_body_site\n"
        + "".join(
            f'{series},{study},{modality},"Acme, Inc.",{BODY_SITE_CODE}\n'
            for study, series, modality in _SERIES
        )
    ),
}

TABLE_NAMES = tuple(_TABLES)

# Not in `_TABLES`: the Fed-BioMed reader does not expose it as a clinical table
IMAGE_MODALITY = (
    "modality_id,series_uid,study_uid,acquisition_parameter_code,"
    "acquisition_parameter_value_code,acquisition_parameter_value_number,"
    "acquisition_parameter_value_unit\n"
    f"M1,{SE1_MR},{S1},{TISSUE_CONTRAST},{T1},,\n"
    f"M2,{SE1_MR},{S1},{SLICE_THICKNESS},,0.25,{CM}\n"
    f"M3,{SE4_MR},{S4},{TISSUE_CONTRAST},{T2},,\n"
    f"M4,{SE4_MR},{S4},{SLICE_THICKNESS},,5,{MM}\n"
    f"M5,{SE1_CT},{S1},IMG1016306,,2.5,COM1000152\n"
    f"M6,{SE_ORPHAN},{_uid(999)},{TISSUE_CONTRAST},{T1},,\n"
)


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
    """Writes a synthetic CDM dataset under `root` (see the module docstring).

    Args:
        root: Dataset root folder (created if needed).
        omit_tables: Tables whose CSV file is not written (`"image_modality"`
            included).
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
    if "image_modality" not in omit_tables:
        (clinical / "image_modality.csv").write_text(
            overrides.get("image_modality", IMAGE_MODALITY), encoding="utf-8"
        )

    imaging = root / "imaging_data"
    shape = SERIES_OK_SHAPE
    write_dicom_series(
        imaging / PATIENT_IMG / STUDY / SERIES_OK,
        patient_id=PATIENT_IMG,
        study_uid=STUDY,
        series_uid=SERIES_OK,
        sizes=[(shape["rows"], shape["cols"])] * shape["slices"],
        **SERIES_OK_RESCALE,
    )
    for patient, study, series, n_slices in _READABLE_SERIES:
        write_dicom_series(
            imaging / patient / study / series,
            patient_id=patient,
            study_uid=study,
            series_uid=series,
            sizes=[(3, 3)] * n_slices,
        )
    write_dicom_series(
        imaging / PATIENT_BROKEN / STUDY_BROKEN / SERIES_BAD_SIZE,
        patient_id=PATIENT_BROKEN,
        study_uid=STUDY_BROKEN,
        series_uid=SERIES_BAD_SIZE,
        sizes=[(4, 5), (4, 6)],
    )
    unreadable = imaging / PATIENT_BROKEN / STUDY_BROKEN / SERIES_UNREADABLE
    unreadable.mkdir(parents=True)
    (unreadable / "notes.txt").write_text("not a DICOM file", encoding="utf-8")

    return root


def add_mr_sequences_patient(root: Path) -> Path:
    """Adds patient P009 to a dataset written by `build_cdm_dataset`.

    Its study S9 holds a T1-weighted and a T2-weighted MR series (both with the MR
    `series_modality` code), each written on disk with 2 slices.

    Returns:
        `root`.
    """
    clinical = root / "clinical_data"
    rows = {
        "patient": f"{PATIENT_MR_SEQ},{MALE},ORG-1,EUCAIM:COM1001087\n",
        "image_study": f"{S9},{PATIENT_MR_SEQ},PR_IMG,2020-09-01,,\n",
        "image_series": "".join(
            f'{series},{S9},{MR_CODE},"Acme, Inc.",{BODY_SITE_CODE}\n'
            for series in (SE9_T1, SE9_T2)
        ),
        "image_modality": (
            f"M91,{SE9_T1},{S9},{TISSUE_CONTRAST},{T1},,\n"
            f"M92,{SE9_T2},{S9},{TISSUE_CONTRAST},{T2},,\n"
        ),
    }
    for table, content in rows.items():
        with open(clinical / f"{table}.csv", "a", encoding="utf-8") as f:
            f.write(content)

    for series in (SE9_T1, SE9_T2):
        write_dicom_series(
            root / "imaging_data" / PATIENT_MR_SEQ / S9 / series,
            patient_id=PATIENT_MR_SEQ,
            study_uid=S9,
            series_uid=series,
            sizes=[(3, 3)] * 2,
        )
    return root
