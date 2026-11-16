import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest
from eucaim_cdm_reader import ParameterFilter
from testsupport import fake_eucaim_cdm as fake

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataset import _mappings
from fedbiomed.common.dataset_controller._eucaim_cdm_controller import (
    CLINICAL_TABLES,
    DROP_DUPLICATE_MODALITY,
    DROP_MISSING_MODALITIES,
    DROP_NO_IMAGING,
    DROP_NO_ORDERING,
    DROP_SEVERAL_CANDIDATES,
    DROP_TIE,
    EucaimCdmController,
)
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import (
    EucaimCdmReader,
    SeriesQuery,
)
from fedbiomed.common.exceptions import FedbiomedError

CT, MR = fake.CT_CODE, fake.MR_CODE


@pytest.fixture
def cdm_root(tmp_path: Path) -> Path:
    return fake.build_cdm_dataset(tmp_path / "cdm")


@pytest.fixture
def controller(cdm_root: Path) -> EucaimCdmController:
    return EucaimCdmController(cdm_root)


def _selected(controller):
    return [(s["patient_id"], s["study_uid"]) for s in controller.samples]


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


def _logged_drops(records):
    """Drop counts by reason, from the last "patient(s) dropped" warning ({} if none)."""
    messages = [
        r.getMessage() for r in records if "patient(s) dropped:" in r.getMessage()
    ]
    if not messages:
        return {}
    counts = messages[-1].split("dropped: ", 1)[1]
    return {k: int(v) for k, v in (item.split("=") for item in counts.split(", "))}


# --- construction and index --------------------------------------------------------------


def test_eucaim_cdm_controller_01_index(controller, cdm_root):
    assert controller.patients == list(fake.PATIENTS)
    assert controller.image_modalities == sorted(
        [CT, MR, fake.PET_CODE, fake.SPECT_CODE, fake.PCCT_CODE]
    )
    assert controller._controller_kwargs == {"root": str(cdm_root.resolve())}
    assert controller.get_dataset_type() == DatasetTypes.EUCAIM_CDM
    # Before select(): one sample per patient, nothing selected yet
    assert len(controller) == len(fake.PATIENTS)
    assert controller.samples is None


def test_eucaim_cdm_controller_02_orphans_are_ignored(controller):
    # The orphan series (study absent) and the orphan study (patient absent) are not
    # indexed: CT is counted once per series of an indexed study.
    assert controller.shape()["series_per_modality"][CT] == 9
    assert fake.S_ORPHAN not in set(controller._studies["study_uid"])
    assert fake.SE_ORPHAN not in set(controller._series["series_uid"])


def test_eucaim_cdm_controller_03_shape_and_types(controller):
    assert controller.shape() == {
        "patients": 8,
        "series_per_modality": {
            MR: 2,
            CT: 9,
            fake.SPECT_CODE: 1,
            fake.PET_CODE: 1,
            fake.PCCT_CODE: 1,
        },
    }
    types = controller.get_types()
    assert types["clinical"] == "dict"
    assert {k: v for k, v in types.items() if k != "clinical"} == {
        code: "DicomVolume" for code in controller.image_modalities
    }


def test_eucaim_cdm_controller_04_no_dicom_read_for_index_and_selection(cdm_root):
    shutil.rmtree(cdm_root / "imaging_data")
    controller = EucaimCdmController(cdm_root)
    controller.shape()
    controller.get_types()
    controller.select([CT, MR])
    assert len(controller) == 1
    # Only get_sample reads images
    with pytest.raises(FedbiomedError, match="no readable DICOM instance"):
        controller.get_sample(0)


def test_eucaim_cdm_controller_05_root_must_exist(tmp_path):
    with pytest.raises(FedbiomedError, match="Path does not exist"):
        EucaimCdmController(tmp_path / "nope")


@pytest.mark.parametrize("name", ["patient", "image_study", "image_series"])
def test_eucaim_cdm_controller_06_missing_index_table_raises(tmp_path, name):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=[name])
    with pytest.raises(FedbiomedError) as exc:
        EucaimCdmController(root)
    assert str(exc.value).startswith(ErrorNumbers.FB634.value)
    assert f"'{name}'" in str(exc.value)


def test_eucaim_cdm_controller_07_other_missing_tables_do_not_block_construction(
    tmp_path,
):
    root = fake.build_cdm_dataset(
        tmp_path / "cdm", omit_tables=["episode", "treatment", "cancer_condition"]
    )
    assert len(EucaimCdmController(root)) == len(fake.PATIENTS)


def test_eucaim_cdm_controller_08_no_patient_raises(tmp_path):
    header = fake._TABLES["patient"].splitlines()[0]
    root = fake.build_cdm_dataset(tmp_path / "cdm", overrides={"patient": header})
    with pytest.raises(FedbiomedError, match="no patient can be indexed"):
        EucaimCdmController(root)


def test_eucaim_cdm_controller_09_ambiguous_identifiers_are_ignored(tmp_path):
    patient = fake._TABLES["patient"]
    study = fake._TABLES["image_study"]
    series = fake._TABLES["image_series"]
    root = fake.build_cdm_dataset(
        tmp_path / "cdm",
        overrides={
            # P002 twice
            "patient": patient + patient.splitlines()[2] + "\n",
            # S3A twice
            "image_study": study
            + next(line for line in study.splitlines() if line.startswith(fake.S3A))
            + "\n",
            # SE1_MR twice
            "image_series": series
            + next(line for line in series.splitlines() if line.startswith(fake.SE1_MR))
            + "\n",
        },
    )
    controller = EucaimCdmController(root)

    assert "P002" not in controller.patients
    assert fake.S3A not in set(controller._studies["study_uid"])
    # Series of an ignored study are orphans
    assert fake.SE3A_CT not in set(controller._series["series_uid"])
    assert fake.SE1_MR not in set(controller._series["series_uid"])

    # P003 now has a single CT study
    controller.select([CT], "unique")
    assert _selected(controller) == [("P001", fake.S1), ("P003", fake.S3B)]


# --- select(): timepoints ----------------------------------------------------------------


def test_eucaim_cdm_controller_10_get_sample_requires_select(controller):
    with pytest.raises(FedbiomedError, match="must be called before"):
        controller.get_sample(0)


@pytest.mark.parametrize("policy", ["unique", "earliest", "latest"])
def test_eucaim_cdm_controller_11_clinical_only_keeps_every_patient(
    controller, fed_records, policy
):
    controller.select([], policy, ["patient"])
    assert _selected(controller) == [(p, None) for p in fake.PATIENTS]
    assert _logged_drops(fed_records) == {}
    assert len(controller) == len(fake.PATIENTS)


@pytest.mark.parametrize(
    "modalities, policy, expected, dropped",
    [
        (
            [CT],
            "unique",
            [("P001", fake.S1)],
            {
                DROP_NO_IMAGING: 1,  # P002
                DROP_SEVERAL_CANDIDATES: 3,  # P003, P005, P006
                DROP_DUPLICATE_MODALITY: 1,  # P004
                DROP_MISSING_MODALITIES: 2,  # P007, P008
            },
        ),
        (
            [CT],
            "earliest",
            # P003: S3B is undated, so offsets (same unit) order the candidates
            [("P001", fake.S1), ("P003", fake.S3A)],
            {
                DROP_NO_IMAGING: 1,
                DROP_DUPLICATE_MODALITY: 1,
                DROP_NO_ORDERING: 1,  # P005: undated, offsets in different units
                DROP_TIE: 1,  # P006: same date
                DROP_MISSING_MODALITIES: 2,
            },
        ),
        (
            [CT],
            "latest",
            [("P001", fake.S1), ("P003", fake.S3B)],
            {
                DROP_NO_IMAGING: 1,
                DROP_DUPLICATE_MODALITY: 1,
                DROP_NO_ORDERING: 1,
                DROP_TIE: 1,
                DROP_MISSING_MODALITIES: 2,
            },
        ),
        (
            [MR],
            "unique",
            [("P001", fake.S1), ("P004", fake.S4)],
            {DROP_NO_IMAGING: 1, DROP_MISSING_MODALITIES: 5},
        ),
        (
            # Same study for both modalities: S4 has CT twice, P003 has no MR
            [CT, MR],
            "latest",
            [("P001", fake.S1)],
            {
                DROP_NO_IMAGING: 1,
                DROP_DUPLICATE_MODALITY: 1,
                DROP_MISSING_MODALITIES: 5,
            },
        ),
    ],
)
def test_eucaim_cdm_controller_12_timepoint_selection(
    controller, fed_records, modalities, policy, expected, dropped
):
    controller.select(modalities, policy)
    assert _selected(controller) == expected
    assert _logged_drops(fed_records) == dropped
    assert sum(dropped.values()) + len(expected) == len(fake.PATIENTS)


def test_eucaim_cdm_controller_13_selected_series(controller):
    controller.select([MR, CT])
    assert controller.samples == [
        {
            "patient_id": "P001",
            "study_uid": fake.S1,
            "series": {MR: fake.SE1_MR, CT: fake.SE1_CT},
        }
    ]


def test_eucaim_cdm_controller_14_modalities_as_str(controller):
    controller.select(MR)
    assert _selected(controller) == [("P001", fake.S1), ("P004", fake.S4)]


def test_eucaim_cdm_controller_15_offsets_need_the_unit_column(tmp_path, fed_records):
    """Without the unit column, offsets cannot order undated candidates."""
    study = fake._TABLES["image_study"].splitlines()
    rows = [",".join(line.split(",")[:5]) for line in study]  # drop study_offset_unit
    root = fake.build_cdm_dataset(
        tmp_path / "cdm", overrides={"image_study": "\n".join(rows) + "\n"}
    )
    controller = EucaimCdmController(root)
    controller.select([CT], "earliest")
    assert _selected(controller) == [("P001", fake.S1)]
    assert _logged_drops(fed_records)[DROP_NO_ORDERING] == 2  # P003, P005


def test_eucaim_cdm_controller_16_no_date_column(tmp_path, fed_records):
    study = fake._TABLES["image_study"].splitlines()
    rows = [
        ",".join(v for i, v in enumerate(line.split(",")) if i != 3) for line in study
    ]
    root = fake.build_cdm_dataset(
        tmp_path / "cdm", overrides={"image_study": "\n".join(rows) + "\n"}
    )
    controller = EucaimCdmController(root)
    controller.select([CT], "latest")
    # P003 still ordered by offsets; P006 has neither dates nor offsets
    assert _selected(controller) == [("P001", fake.S1), ("P003", fake.S3B)]
    assert _logged_drops(fed_records)[DROP_NO_ORDERING] == 2  # P005, P006


@pytest.mark.parametrize("policy, expected", [("earliest", 0), ("latest", 1)])
def test_eucaim_cdm_controller_17_ordered_by_dates(tmp_path, policy, expected):
    """Dated candidates are ordered by date, even when their offsets disagree."""
    study = fake._TABLES["image_study"]
    # P006: S6B one day after S6A (it was a tie); offsets in the opposite order
    study = study.replace(
        f"{fake.S6A},P006,PR_IMG,2020-05-05,,",
        f"{fake.S6A},P006,PR_IMG,2020-05-05,9,{fake.DAY}",
    )
    study = study.replace(
        f"{fake.S6B},P006,PR_IMG,2020-05-05,,",
        f"{fake.S6B},P006,PR_IMG,2020-05-06,1,{fake.DAY}",
    )
    root = fake.build_cdm_dataset(tmp_path / "cdm", overrides={"image_study": study})
    controller = EucaimCdmController(root)
    controller.select([CT], policy)
    assert ("P006", (fake.S6A, fake.S6B)[expected]) in _selected(controller)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"timepoint_policy": "first"}, "timepoint_policy"),
        ({"tables": ["episode_event"]}, "unsupported table"),
        ({"tables": ["image_modality"]}, "unsupported table"),
        ({"tables": 3}, "iterable of str"),
        ({"modalities": [CT, 3]}, "non-empty str"),
        ({"modalities": [""]}, "non-empty str"),
    ],
)
def test_eucaim_cdm_controller_18_select_bad_arguments(controller, kwargs, match):
    with pytest.raises(FedbiomedError, match=match):
        controller.select(**kwargs)


def test_eucaim_cdm_controller_19_no_sample_left_keeps_previous_selection(controller):
    controller.select([MR])
    with pytest.raises(FedbiomedError, match="no sample left") as exc:
        controller.select(["EUCAIM:IMG1000024"])
    assert DROP_MISSING_MODALITIES in str(exc.value)
    assert _selected(controller) == [("P001", fake.S1), ("P004", fake.S4)]


def test_eucaim_cdm_controller_20_select_is_reentrant(controller, fed_records):
    controller.select([CT], "earliest", ["treatment"])
    controller.select([MR], "unique", ["patient"])
    assert _selected(controller) == [("P001", fake.S1), ("P004", fake.S4)]
    assert _logged_drops(fed_records) == {
        DROP_NO_IMAGING: 1,
        DROP_MISSING_MODALITIES: 5,
    }
    sample = controller.get_sample(1)
    assert list(sample["clinical"]) == ["patient"]


# --- get_sample(): clinical rows ---------------------------------------------------------


def test_eucaim_cdm_controller_21_rows_clinical_only(controller):
    controller.select([], tables=CLINICAL_TABLES)
    by_patient = {p: i for i, p in enumerate(fake.PATIENTS)}

    sample = controller.get_sample(by_patient["P001"])
    assert set(sample) == {"clinical"}  # no image modality
    clinical = sample["clinical"]
    assert list(clinical) == list(CLINICAL_TABLES)
    assert list(clinical["patient"]["patient_id"]) == ["P001"]
    assert list(clinical["treatment"]["treatment_id"]) == ["T1", "T2"]
    assert list(clinical["procedure"]["procedure_id"]) == ["PR1", "PR_IMG"]

    # All studies and all their series
    clinical = controller.get_sample(by_patient["P003"])["clinical"]
    assert list(clinical["image_study"]["study_uid"]) == [fake.S3A, fake.S3B]
    assert list(clinical["image_series"]["series_uid"]) == [fake.SE3A_CT, fake.SE3B_CT]
    clinical = controller.get_sample(by_patient["P004"])["clinical"]
    assert list(clinical["image_series"]["series_uid"]) == [
        fake.SE4_CT1,
        fake.SE4_CT2,
        fake.SE4_MR,
    ]


def test_eucaim_cdm_controller_22_rows_for_patient_without_rows(controller):
    controller.select([], tables=["treatment", "image_study", "image_series"])
    clinical = controller.get_sample(fake.PATIENTS.index("P002"))["clinical"]
    for name, df in clinical.items():
        assert df.empty
        # Columns and dtypes as in the table
        assert "patient_id" in df.columns or name == "image_series"
    assert isinstance(clinical["treatment"]["patient_id"].dtype, pd.StringDtype)


def test_eucaim_cdm_controller_23_rows_of_selected_study(controller):
    controller.select([CT], "earliest", ["image_study", "image_series", "patient"])
    assert _selected(controller)[1] == ("P003", fake.S3A)
    clinical = controller.get_sample(1)["clinical"]
    assert list(clinical["image_study"]["study_uid"]) == [fake.S3A]
    assert list(clinical["image_series"]["series_uid"]) == [fake.SE3A_CT]
    assert list(clinical["patient"]["patient_id"]) == ["P003"]


def test_eucaim_cdm_controller_24_rows_of_selected_series_only(controller):
    # S1 also holds the CT series, which is not requested
    controller.select([MR], tables=["image_series"])
    clinical = controller.get_sample(0)["clinical"]
    assert list(clinical["image_series"]["series_uid"]) == [fake.SE1_MR]

    # Several modalities: one row per selected series, in table order
    controller.select([MR, CT], tables=["image_series"])
    series_rows = controller.get_sample(0)["clinical"]["image_series"]
    assert list(series_rows["series_uid"]) == [fake.SE1_CT, fake.SE1_MR]


def test_eucaim_cdm_controller_25_rows_are_copies(controller):
    controller.select([], tables=["treatment"])
    rows = controller.get_sample(0)["clinical"]["treatment"]
    rows.loc[:, "treatment_id"] = "tampered"
    rows.drop(index=rows.index, inplace=True)
    assert list(controller.get_sample(0)["clinical"]["treatment"]["treatment_id"]) == [
        "T1",
        "T2",
    ]


def test_eucaim_cdm_controller_26_missing_optional_table(tmp_path):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=["radiotherapy"])
    controller = EucaimCdmController(root)
    controller.select([], tables=["radiotherapy", "treatment"])
    assert controller.missing_tables == ["radiotherapy"]
    clinical = controller.get_sample(0)["clinical"]
    assert clinical["radiotherapy"].empty
    assert len(clinical["radiotherapy"].columns) == 0
    assert len(clinical["treatment"]) == 2


def test_eucaim_cdm_controller_27_missing_mandatory_table_on_select(tmp_path):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=["episode"])
    controller = EucaimCdmController(root)
    controller.select([MR])
    with pytest.raises(FedbiomedError, match="mandatory EUCAIM CDM table 'episode'"):
        controller.select([], tables=["episode"])
    # Previous selection kept
    assert _selected(controller) == [("P001", fake.S1), ("P004", fake.S4)]


def test_eucaim_cdm_controller_28_index_out_of_range(controller):
    controller.select([MR])
    with pytest.raises(IndexError):
        controller.get_sample(2)


# --- get_sample(): images ----------------------------------------------------------------


def test_eucaim_cdm_controller_29_volumes(controller):
    controller.select([CT, MR], tables=["patient"])
    sample = controller.get_sample(0)
    assert set(sample) == {"clinical", CT, MR}
    shape = fake.SERIES_OK_SHAPE
    assert sample[CT].array.shape == (shape["cols"], shape["rows"], shape["slices"])
    assert sample[MR].array.shape == (3, 3, 2)


def test_eucaim_cdm_controller_30_volume_of_selected_study(controller, mocker):
    read_series = mocker.spy(EucaimCdmReader, "read_series")
    controller.select([CT], "latest")
    controller.get_sample(1)
    args = read_series.call_args.args
    assert args[1:] == ("P003", fake.S3B, fake.SE3B_CT)


def test_eucaim_cdm_controller_31_unreadable_series_names_the_series(controller):
    controller.select([fake.PCCT_CODE])
    assert _selected(controller) == [(fake.PATIENT_NO_DIR, fake.STUDY_NO_DIR)]
    with pytest.raises(FedbiomedError) as exc:
        controller.get_sample(0)
    message = str(exc.value)
    assert message.startswith(ErrorNumbers.FB634.value)
    for part in (fake.PATIENT_NO_DIR, fake.STUDY_NO_DIR, fake.SERIES_NO_DIR):
        assert part in message


# --- registry round trip -----------------------------------------------------------------


def test_eucaim_cdm_controller_32_get_controller_round_trip(cdm_root, monkeypatch):
    monkeypatch.setitem(
        _mappings.REGISTRY_CONTROLLERS,
        DatasetTypes.EUCAIM_CDM,
        (EucaimCdmController, None),
    )
    # As the dataset manager does at registration: unknown keys and None are dropped
    first = _mappings.get_controller(
        "eucaim-cdm", {"root": str(cdm_root), "tabular_file": None, "dlp": None}
    )
    assert isinstance(first, EucaimCdmController)

    # As done when a stored dataset is reloaded from its parameters
    second = _mappings.get_controller("eucaim-cdm", first._controller_kwargs)
    assert second._controller_kwargs == first._controller_kwargs
    assert second.patients == first.patients


# --- series queries ----------------------------------------------------------------------

T1_QUERY = SeriesQuery(MR, ParameterFilter(fake.TISSUE_CONTRAST, value_code=fake.T1))
T2_QUERY = SeriesQuery(MR, ParameterFilter(fake.TISSUE_CONTRAST, value_code=fake.T2))


@pytest.fixture
def mr_seq_controller(cdm_root: Path) -> EucaimCdmController:
    return EucaimCdmController(fake.add_mr_sequences_patient(cdm_root))


def test_eucaim_cdm_controller_33_queries_tell_same_modality_series_apart(
    mr_seq_controller, fed_records
):
    # As plain MR, S9 holds the modality twice and is not a candidate
    mr_seq_controller.select([MR])
    assert _selected(mr_seq_controller) == [("P001", fake.S1), ("P004", fake.S4)]
    assert _logged_drops(fed_records)[DROP_DUPLICATE_MODALITY] == 1

    mr_seq_controller.select({"t2": T2_QUERY, "t1": T1_QUERY})
    assert mr_seq_controller.samples == [
        {
            "patient_id": fake.PATIENT_MR_SEQ,
            "study_uid": fake.S9,
            "series": {"t1": fake.SE9_T1, "t2": fake.SE9_T2},
        }
    ]
    sample = mr_seq_controller.get_sample(0)
    assert list(sample) == ["t1", "t2"]
    assert sample["t1"].array.shape == (3, 3, 2)


def test_eucaim_cdm_controller_34_query_one_series_for_two_modalities(controller):
    controller.select({"mr": SeriesQuery(MR), "t1": T1_QUERY})
    assert controller.samples == [
        {
            "patient_id": "P001",
            "study_uid": fake.S1,
            "series": {"mr": fake.SE1_MR, "t1": fake.SE1_MR},
        }
    ]


def test_eucaim_cdm_controller_35_query_numeric_filter(controller):
    # SE1_MR: 0.25 cm, converted to 2.5 mm; SE4_MR: 5 mm
    thin = ParameterFilter(fake.SLICE_THICKNESS, max_value=3, unit=fake.MM)
    controller.select({"thin_mr": SeriesQuery(MR, thin)})
    assert _selected(controller) == [("P001", fake.S1)]


def test_eucaim_cdm_controller_36_query_ignores_unindexed_series(controller, mocker):
    # The reader finds the orphan series too, the controller never selects it
    find_series = mocker.spy(EucaimCdmReader, "find_series")
    controller.select({"t1": SeriesQuery(parameters=T1_QUERY.parameters)})
    assert fake.SE_ORPHAN in find_series.spy_return
    assert _selected(controller) == [("P001", fake.S1)]


def test_eucaim_cdm_controller_37_bare_codes_are_queries(controller, mocker):
    find_series = mocker.spy(EucaimCdmReader, "find_series")
    controller.select([CT, MR])
    queries = [call.args[1] for call in find_series.call_args_list]
    assert queries == [SeriesQuery(MR), SeriesQuery(CT)]  # sorted by name


@pytest.mark.parametrize(
    "modalities, match",
    [
        ({"t1": MR}, "expected a SeriesQuery"),
        ({"x": SeriesQuery("EUCAIM:IMG1000022")}, "not a `series_modality` code"),
        (["IMG1000038", "MR"], "not a `series_modality` code"),
        ({"": T1_QUERY}, "non-empty str"),
    ],
)
def test_eucaim_cdm_controller_38_select_bad_queries(controller, modalities, match):
    controller.select([MR])
    with pytest.raises(FedbiomedError, match=match):
        controller.select(modalities)
    assert _selected(controller) == [("P001", fake.S1), ("P004", fake.S4)]


def test_eucaim_cdm_controller_39_query_without_image_modality(tmp_path):
    root = fake.build_cdm_dataset(tmp_path / "cdm", omit_tables=["image_modality"])
    controller = EucaimCdmController(root)
    controller.select([MR])  # modality only: no need for image_modality
    with pytest.raises(FedbiomedError, match="no sample left") as exc:
        controller.select({"t1": T1_QUERY})
    assert DROP_MISSING_MODALITIES in str(exc.value)
