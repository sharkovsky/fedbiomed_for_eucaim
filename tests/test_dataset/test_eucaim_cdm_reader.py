import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from testsupport import fake_eucaim_cdm as fake

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_reader import _eucaim_cdm_reader
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import EucaimCdmReader
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def cdm_root(tmp_path: Path) -> Path:
    return fake.build_cdm_dataset(tmp_path / "cdm")


@pytest.fixture
def reader(cdm_root: Path) -> EucaimCdmReader:
    return EucaimCdmReader(cdm_root)


# --- construction ---------------------------------------------------------------------


def test_eucaim_cdm_reader_01_accepts_str_and_path(cdm_root):
    assert EucaimCdmReader(str(cdm_root)).root() == cdm_root
    assert EucaimCdmReader(cdm_root).root() == cdm_root


def test_eucaim_cdm_reader_02_root_must_be_a_folder(tmp_path):
    with pytest.raises(FedbiomedError, match="not a folder"):
        EucaimCdmReader(tmp_path / "does-not-exist")

    a_file = tmp_path / "file.csv"
    a_file.write_text("x\n", encoding="utf-8")
    with pytest.raises(FedbiomedError, match="not a folder"):
        EucaimCdmReader(a_file)


def test_eucaim_cdm_reader_03_root_bad_type():
    with pytest.raises(FedbiomedError, match="must be a str or Path"):
        EucaimCdmReader(42)


def test_eucaim_cdm_reader_04_mandatory_and_optional_tables_partition_all_tables():
    mandatory = set(EucaimCdmReader.MANDATORY_TABLES)
    optional = set(EucaimCdmReader.OPTIONAL_TABLES)
    assert optional == {"treatment", "surgical_procedure", "radiotherapy"}
    assert {"patient", "image_study", "image_series"} <= mandatory
    assert not mandatory & optional
    assert mandatory | optional == set(fake.TABLE_NAMES)


# --- table() ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", fake.TABLE_NAMES)
def test_eucaim_cdm_reader_05_every_table_loads(reader, name):
    df = reader.table(name)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_eucaim_cdm_reader_06_table_types(reader):
    patients = reader.table("patient")
    assert list(patients["patient_id"]) == [fake.PATIENT_IMG, fake.PATIENT_NO_IMG]
    assert isinstance(patients["patient_id"].dtype, pd.StringDtype)

    conditions = reader.table("cancer_condition")
    assert isinstance(
        conditions["cancer_condition_age_at_diagnosis"].dtype, pd.Float64Dtype
    )
    assert pd.api.types.is_datetime64_any_dtype(
        conditions["cancer_condition_asserted_date"]
    )
    assert conditions["cancer_condition_asserted_date"].iloc[0] == pd.Timestamp(
        "2020-01-15"
    )


def test_eucaim_cdm_reader_07_table_returns_a_copy(reader):
    first = reader.table("patient")
    first.loc[:, "patient_id"] = "tampered"
    first.drop(index=first.index, inplace=True)

    second = reader.table("patient")
    assert list(second["patient_id"]) == [fake.PATIENT_IMG, fake.PATIENT_NO_IMG]


def test_eucaim_cdm_reader_08_missing_optional_column_is_absent(reader):
    treatment = reader.table("treatment")
    assert "treatment_type" in treatment.columns
    assert "start_offset_from_diagnosis" not in treatment.columns
    assert "start_offset_unit" not in treatment.columns


@pytest.mark.parametrize("name", EucaimCdmReader.OPTIONAL_TABLES)
def test_eucaim_cdm_reader_09_missing_optional_table_is_none(tmp_path, name):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=[name])
    reader = EucaimCdmReader(root)
    assert reader.table(name) is None
    # Other tables are unaffected
    assert reader.table("patient") is not None


@pytest.mark.parametrize("name", EucaimCdmReader.MANDATORY_TABLES)
def test_eucaim_cdm_reader_10_missing_mandatory_table_raises(tmp_path, name):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=[name])
    reader = EucaimCdmReader(root)
    with pytest.raises(FedbiomedError, match=f"mandatory EUCAIM CDM table '{name}'"):
        reader.table(name)


def test_eucaim_cdm_reader_11_unsupported_table_raises(reader):
    for name in ("image_modality", "segment", "clinical_mandatory_view", "nope"):
        with pytest.raises(FedbiomedError, match="unsupported EUCAIM CDM table"):
            reader.table(name)


def test_eucaim_cdm_reader_12_missing_required_column_raises(tmp_path):
    root = fake.build_cdm_dataset(
        tmp_path / "cdm",
        overrides={"treatment": "treatment_id,patient_id\nT1,P001\n"},
    )
    reader = EucaimCdmReader(root)
    with pytest.raises(
        FedbiomedError, match="cannot load EUCAIM CDM table 'treatment'"
    ):
        reader.table("treatment")


def test_eucaim_cdm_reader_13_empty_csv_raises(tmp_path):
    root = fake.build_cdm_dataset(tmp_path / "cdm", overrides={"episode": ""})
    reader = EucaimCdmReader(root)
    with pytest.raises(FedbiomedError, match="cannot load EUCAIM CDM table 'episode'"):
        reader.table("episode")


# --- field_spec() -----------------------------------------------------------------------


def test_eucaim_cdm_reader_14_field_spec():
    spec = EucaimCdmReader.field_spec("image_series", "series_modality")
    assert spec.name == "series_modality"
    assert spec.code_list is not None
    assert fake.CT_CODE.removeprefix("EUCAIM:") in spec.code_list

    spec = EucaimCdmReader.field_spec(
        "cancer_condition", "cancer_condition_asserted_date"
    )
    assert spec.is_date


def test_eucaim_cdm_reader_15_field_spec_errors():
    with pytest.raises(FedbiomedError, match="unsupported EUCAIM CDM table"):
        EucaimCdmReader.field_spec("segment", "segment_id")
    with pytest.raises(FedbiomedError, match="has no column 'nope'"):
        EucaimCdmReader.field_spec("patient", "nope")


# --- read_series() ----------------------------------------------------------------------


def test_eucaim_cdm_reader_16_read_series_default(reader):
    vol = reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_OK)

    shape = fake.SERIES_OK_SHAPE
    slope = fake.SERIES_OK_RESCALE["slope"]
    intercept = fake.SERIES_OK_RESCALE["intercept"]

    # (X, Y, Z) = (cols, rows, slices)
    assert vol.array.shape == (shape["cols"], shape["rows"], shape["slices"])
    # Slices were written in reverse file-name order: sorted by position here.
    # Slice k holds raw value k + 1, rescaled.
    for k in range(shape["slices"]):
        np.testing.assert_array_equal(vol.array[:, :, k], (k + 1) * slope + intercept)
    assert vol.affine is not None and vol.affine.shape == (4, 4)
    np.testing.assert_allclose(vol.spacing, (0.7, 0.5, 2.5))


def test_eucaim_cdm_reader_17_read_series_options(reader):
    vol = reader.read_series(
        fake.PATIENT_IMG,
        fake.STUDY,
        fake.SERIES_OK,
        ras=False,
        xyz_order=False,
        rescale=False,
    )
    shape = fake.SERIES_OK_SHAPE
    # (Z, Y, X), raw values
    assert vol.array.shape == (shape["slices"], shape["rows"], shape["cols"])
    for k in range(shape["slices"]):
        np.testing.assert_array_equal(vol.array[k], k + 1)


@pytest.mark.parametrize(
    "patient_id, study_uid, series_uid",
    [
        # indexed, but no folder on disk
        (fake.PATIENT_IMG, fake.STUDY, fake.SERIES_NO_DIR),
        # folder holds no readable DICOM file
        (fake.PATIENT_IMG, fake.STUDY, fake.SERIES_UNREADABLE),
        # study does not belong to that patient
        (fake.PATIENT_NO_IMG, fake.STUDY, fake.SERIES_OK),
        # series does not belong to that study
        (fake.PATIENT_IMG, "1.2.3", fake.SERIES_OK),
        # unknown series
        (fake.PATIENT_IMG, fake.STUDY, "1.2.3"),
    ],
)
def test_eucaim_cdm_reader_18_read_series_nothing_to_read(
    reader, patient_id, study_uid, series_uid
):
    with pytest.raises(FedbiomedError, match="no readable DICOM instance") as exc:
        reader.read_series(patient_id, study_uid, series_uid)
    assert patient_id in str(exc.value)
    assert study_uid in str(exc.value)
    assert series_uid in str(exc.value)


def test_eucaim_cdm_reader_19_read_series_inconsistent_slices(reader):
    with pytest.raises(FedbiomedError, match="cannot build a volume"):
        reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_BAD_SIZE)


@pytest.mark.parametrize("name", ["image_study", "image_series"])
def test_eucaim_cdm_reader_20_read_series_missing_index_table(tmp_path, name):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=[name])
    reader = EucaimCdmReader(root)
    with pytest.raises(FedbiomedError, match="index table missing"):
        reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_OK)


def test_eucaim_cdm_reader_21_errors_use_eucaim_error_code(reader):
    with pytest.raises(FedbiomedError) as exc:
        reader.table("nope")
    assert str(exc.value).startswith(ErrorNumbers.FB634.value)


# --- library logs and warnings reach the Fed-BioMed logger ------------------------------


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def fed_records():
    """Records reaching the Fed-BioMed logger's handlers, at the default WARNING level."""
    fed_logger = logging.getLogger("fedbiomed")
    previous_level = fed_logger.level
    fed_logger.setLevel(logging.WARNING)
    handler = _ListHandler()
    fed_logger.addHandler(handler)
    yield handler.records
    fed_logger.removeHandler(handler)
    fed_logger.setLevel(previous_level)


def _library_records(records):
    return [r for r in records if r.name.startswith("eucaim_cdm_reader")]


def test_eucaim_cdm_reader_22_library_log_reaches_fedbiomed_logger(reader, fed_records):
    with pytest.raises(FedbiomedError):
        reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_NO_DIR)

    records = _library_records(fed_records)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.name == "eucaim_cdm_reader.reader"
    assert "DICOM series directory not found" in record.getMessage()
    # Never forwarded to a researcher by the gRPC handler
    assert not hasattr(record, "broadcast")
    assert not hasattr(record, "researcher_id")


def test_eucaim_cdm_reader_23_library_log_level_follows_fedbiomed(fed_records):
    lib_logger = logging.getLogger("eucaim_cdm_reader.reader")

    lib_logger.debug("hidden at WARNING")
    assert _library_records(fed_records) == []

    logging.getLogger("fedbiomed").setLevel(logging.DEBUG)
    lib_logger.debug("shown at DEBUG")
    assert [r.getMessage() for r in _library_records(fed_records)] == ["shown at DEBUG"]


def test_eucaim_cdm_reader_24_library_log_not_duplicated_to_root(
    reader, fed_records, caplog
):
    # caplog listens on the root logger
    caplog.set_level(logging.DEBUG)
    with pytest.raises(FedbiomedError):
        reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_NO_DIR)
    assert _library_records(fed_records)
    assert _library_records(caplog.records) == []


def test_eucaim_cdm_reader_25_bridge_installed_once():
    _eucaim_cdm_reader._bridge_library_logging()
    _eucaim_cdm_reader._bridge_library_logging()
    bridges = [
        h
        for h in logging.getLogger("eucaim_cdm_reader").handlers
        if isinstance(h, _eucaim_cdm_reader._FedLoggerBridge)
    ]
    assert len(bridges) == 1


def _warn_then(result=None, error=None):
    def fake_series_to_volume(datasets, **kwargs):
        warnings.warn("Non-uniform slice spacing detected", UserWarning, stacklevel=2)
        if error is not None:
            raise error
        return result

    return fake_series_to_volume


@pytest.mark.parametrize("fails", [False, True])
def test_eucaim_cdm_reader_26_volume_warnings_are_logged(
    reader, fed_records, monkeypatch, fails
):
    sentinel = object()
    monkeypatch.setattr(
        _eucaim_cdm_reader,
        "series_to_volume",
        _warn_then(result=sentinel, error=ValueError("boom") if fails else None),
    )

    with warnings.catch_warnings(record=True) as escaped:
        warnings.simplefilter("always")
        if fails:
            with pytest.raises(FedbiomedError, match="boom"):
                reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_OK)
        else:
            assert (
                reader.read_series(fake.PATIENT_IMG, fake.STUDY, fake.SERIES_OK)
                is sentinel
            )

    # Captured, not left to the warnings machinery...
    assert escaped == []
    # ... and logged once, with the series it concerns
    records = _library_records(fed_records)
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("UserWarning for ")
    assert fake.SERIES_OK in message
    assert "Non-uniform slice spacing detected" in message
    assert not hasattr(records[0], "broadcast")


def test_eucaim_cdm_reader_27_real_volume_warning_is_logged(tmp_path, fed_records):
    """A series without geometry tags makes `series_to_volume` warn (affine is None)."""
    root = fake.build_cdm_dataset(tmp_path / "cdm")
    series_dir = root / "imaging_data" / fake.PATIENT_IMG / fake.STUDY / fake.SERIES_OK

    for path in series_dir.iterdir():
        ds = pydicom.dcmread(path)
        del ds.ImagePositionPatient
        del ds.ImageOrientationPatient
        ds.save_as(path, enforce_file_format=True)

    vol = EucaimCdmReader(root).read_series(
        fake.PATIENT_IMG, fake.STUDY, fake.SERIES_OK
    )
    assert vol.affine is None
    messages = [r.getMessage() for r in _library_records(fed_records)]
    assert any("geometry tags" in m and fake.SERIES_OK in m for m in messages)
