import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from eucaim_cdm_reader import ParameterFilter, schema
from testsupport import fake_eucaim_cdm as fake

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_reader import _eucaim_cdm_reader
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import (
    EucaimCdmReader,
    SeriesQuery,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


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
    assert list(patients["patient_id"]) == list(fake.PATIENTS)
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
    assert list(second["patient_id"]) == list(fake.PATIENTS)


def test_eucaim_cdm_reader_08_missing_optional_column_is_absent(reader):
    procedure = reader.table("procedure")
    assert "procedure_code" in procedure.columns
    assert "procedure_offset_from_diagnosis" not in procedure.columns
    assert "procedure_offset_unit" not in procedure.columns


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
    assert spec.kind == schema.CODE
    assert fake.CT_CODE.removeprefix("EUCAIM:") in spec.codes

    spec = EucaimCdmReader.field_spec(
        "cancer_condition", "cancer_condition_asserted_date"
    )
    assert spec.kind == schema.DATE


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
        (fake.PATIENT_NO_DIR, fake.STUDY_NO_DIR, fake.SERIES_NO_DIR),
        # folder holds no readable DICOM file
        (fake.PATIENT_BROKEN, fake.STUDY_BROKEN, fake.SERIES_UNREADABLE),
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
        reader.read_series(fake.PATIENT_BROKEN, fake.STUDY_BROKEN, fake.SERIES_BAD_SIZE)


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
        reader.read_series(fake.PATIENT_NO_DIR, fake.STUDY_NO_DIR, fake.SERIES_NO_DIR)

    records = _library_records(fed_records)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.name == "eucaim_cdm_reader.reader"
    assert "DICOM series folder not found" in record.getMessage()
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
        reader.read_series(fake.PATIENT_NO_DIR, fake.STUDY_NO_DIR, fake.SERIES_NO_DIR)
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


# --- series queries ----------------------------------------------------------------------

T1_FILTER = ParameterFilter(fake.TISSUE_CONTRAST, value_code=fake.T1)


def test_eucaim_cdm_reader_28_series_query():
    query = SeriesQuery(fake.MR_CODE, T1_FILTER)
    assert query.parameters == (T1_FILTER,)
    assert SeriesQuery(parameters=[T1_FILTER]).modality is None
    # Frozen and hashable, so it can be compared and stored
    assert query == SeriesQuery(fake.MR_CODE, [T1_FILTER])
    assert hash(query) == hash(SeriesQuery(fake.MR_CODE, (T1_FILTER,)))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"modality": ""}, "`modality` must be a non-empty str"),
        ({"modality": 3}, "`modality` must be a non-empty str"),
        ({"modality": fake.MR_CODE, "parameters": "x"}, "`parameters` must be a"),
        ({"modality": fake.MR_CODE, "parameters": {}}, "`parameters` must be a"),
        ({"modality": fake.MR_CODE, "parameters": [3]}, "only contain ParameterFilter"),
        ({"modality": fake.MR_CODE, "include_subclasses": 1}, "must be a bool"),
        ({}, "needs a modality, a parameter, or both"),
    ],
)
def test_eucaim_cdm_reader_29_series_query_bad_arguments(kwargs, match):
    with pytest.raises(FedbiomedValueError, match=match) as exc:
        SeriesQuery(**kwargs)
    assert str(exc.value).startswith(ErrorNumbers.FB634.value)


@pytest.mark.parametrize(
    "query",
    [
        SeriesQuery(fake.MR_CODE),
        SeriesQuery("IMG1000038"),  # prefix is optional
        SeriesQuery(fake.MR_CODE, [T1_FILTER, ParameterFilter("IMG1016306")]),
        SeriesQuery(
            parameters=ParameterFilter(
                fake.SLICE_THICKNESS, min_value=1, max_value=3, unit=fake.MM
            )
        ),
    ],
)
def test_eucaim_cdm_reader_30_check_series_query_ok(query):
    EucaimCdmReader.check_series_query(query)


@pytest.mark.parametrize(
    "query, match",
    [
        ("EUCAIM:IMG1000038", "expected a SeriesQuery"),
        (SeriesQuery("EUCAIM:IMG1000022"), "not a `series_modality` code"),
        (SeriesQuery(parameters=ParameterFilter("EUCAIM:IMG1000038")), "code="),
        (
            SeriesQuery(
                parameters=ParameterFilter(
                    fake.TISSUE_CONTRAST, value_code="IMG1000038"
                )
            ),
            "value_code=",
        ),
        (
            SeriesQuery(
                parameters=ParameterFilter(
                    fake.SLICE_THICKNESS, max_value=3, unit=fake.DAY
                )
            ),
            "unit=",
        ),
    ],
)
def test_eucaim_cdm_reader_31_check_series_query_errors(query, match):
    with pytest.raises(FedbiomedValueError, match=match) as exc:
        EucaimCdmReader.check_series_query(query)
    assert str(exc.value).startswith(ErrorNumbers.FB634.value)


def test_eucaim_cdm_reader_32_check_series_query_subclasses_need_rdflib(monkeypatch):
    # A None entry makes `import rdflib` fail, whether rdflib is installed or not
    monkeypatch.setitem(sys.modules, "rdflib", None)
    from eucaim_cdm_reader import ontology

    if getattr(ontology, "_index", None) is not None:  # pragma: no cover
        pytest.skip("ontology already loaded in this process")
    with pytest.raises(FedbiomedValueError, match="rdflib"):
        EucaimCdmReader.check_series_query(
            SeriesQuery(fake.MR_CODE, include_subclasses=True)
        )


@pytest.mark.parametrize(
    "query, expected",
    [
        (SeriesQuery(fake.MR_CODE), {fake.SE1_MR, fake.SE4_MR}),
        (SeriesQuery("IMG1000038"), {fake.SE1_MR, fake.SE4_MR}),
        # The reader does not index: the orphan series matches too
        (SeriesQuery(parameters=T1_FILTER), {fake.SE1_MR, fake.SE_ORPHAN}),
        (SeriesQuery(fake.MR_CODE, T1_FILTER), {fake.SE1_MR}),
        # 0.25 cm and 2.5 mm (stored without prefix) are both <= 3 mm; 5 mm is not
        (
            SeriesQuery(
                parameters=ParameterFilter("IMG1016306", max_value=3, unit="COM1000152")
            ),
            {fake.SE1_MR, fake.SE1_CT},
        ),
        (SeriesQuery(fake.PET_CODE, T1_FILTER), set()),
    ],
)
def test_eucaim_cdm_reader_33_find_series(reader, query, expected):
    assert reader.find_series(query) == expected


def test_eucaim_cdm_reader_34_find_series_without_image_modality(tmp_path, fed_records):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=["image_modality"])
    reader = EucaimCdmReader(root)

    assert reader.find_series(SeriesQuery(fake.MR_CODE)) == {fake.SE1_MR, fake.SE4_MR}
    assert fed_records == []

    assert reader.find_series(SeriesQuery(fake.MR_CODE, T1_FILTER)) == set()
    messages = [r.getMessage() for r in fed_records]
    assert any("'image_modality' is absent" in m for m in messages)


def test_eucaim_cdm_reader_35_find_series_bad_image_modality(tmp_path):
    root = fake.build_cdm_dataset(
        tmp_path / "cdm",
        overrides={"image_modality": "modality_id,series_uid\nM1,X\n"},
    )
    with pytest.raises(FedbiomedError, match="cannot load EUCAIM CDM table"):
        EucaimCdmReader(root).find_series(SeriesQuery(fake.MR_CODE, T1_FILTER))


def test_eucaim_cdm_reader_36_series_query_with_labels(reader):
    """Codes may be given as their label in the CDM schema, in any letter case."""
    by_label = SeriesQuery(
        "magnetic resonance imaging",
        ParameterFilter("Tissue Contrast", value_code="T1 weighted"),
    )
    EucaimCdmReader.check_series_query(by_label)
    assert reader.find_series(by_label) == reader.find_series(
        SeriesQuery(fake.MR_CODE, T1_FILTER)
    )
    assert reader.find_series(by_label) == {fake.SE1_MR}
