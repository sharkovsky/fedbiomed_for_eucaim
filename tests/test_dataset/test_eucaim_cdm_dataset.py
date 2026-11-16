import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from eucaim_cdm_reader import ParameterFilter
from testsupport import fake_eucaim_cdm as fake

from fedbiomed.common.dataset._eucaim_cdm_dataset import EucaimCdmDataset
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import SeriesQuery
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

CT, MR = fake.CT_CODE, fake.MR_CODE
SEX = "patient.patient_birth_sex"
TREATMENT_TYPE = "treatment.treatment_type"
START = "treatment.start_offset_from_diagnosis"
STUDY_OFFSET = "image_study.study_offset_from_diagnosis"

CLINICAL_ARGS = {
    "input_columns": [SEX, START],
    "vocabularies": {SEX: [fake.MALE, fake.FEMALE]},
    "reducers": {START: "mean"},
}


@pytest.fixture
def cdm_root(tmp_path: Path) -> Path:
    return fake.build_cdm_dataset(tmp_path / "cdm")


def _loaded(root, to_format=DataReturnFormat.TORCH, **kwargs):
    dataset = EucaimCdmDataset(**kwargs)
    dataset.load(root, to_format)
    return dataset


# --- __init__ checks (no data) -----------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"data_modalities": []}, "cannot be empty"),
        ({"data_modalities": {"clinical": 1}}, "str or an iterable"),
        ({"data_modalities": [3]}, "only contain str"),
        ({"data_modalities": "MR"}, "neither 'clinical', a"),
        ({"data_modalities": "IMG1000042"}, "'EUCAIM:' prefix"),
        ({"data_modalities": "EUCAIM:IMG1000022"}, "neither 'clinical', a"),
        ({"data_modalities": "clinical"}, "`input_columns` is required"),
        (
            {"data_modalities": CT, "input_columns": [SEX]},
            "`input_columns` given but 'clinical'",
        ),
        (
            {"data_modalities": CT, "target_modalities": MR, "target_columns": [SEX]},
            "`target_columns` given but 'clinical'",
        ),
        (
            {"data_modalities": CT, "target_transform": lambda x: x},
            "`target_modalities` is None",
        ),
        ({"data_modalities": CT, "timepoint_policy": "first"}, "timepoint_policy"),
        (
            {
                "data_modalities": "clinical",
                "input_columns": [SEX],
                "vocabularies": {TREATMENT_TYPE: [fake.CHEMO]},
            },
            "`vocabularies` given for unselected",
        ),
        (
            {
                "data_modalities": "clinical",
                "input_columns": [SEX],
                "reducers": {START: "max"},
            },
            "`reducers` given for unselected",
        ),
        (
            {"data_modalities": CT, "transform": {MR: lambda x: x}},
            "unexpected keys in transform",
        ),
        ({"data_modalities": CT, "transform": {CT: 3}}, "map modalities to callables"),
        ({"data_modalities": CT, "transform": 3}, "unexpected type for transform"),
        # From the encoding plan
        (
            {"data_modalities": "clinical", "input_columns": [START]},
            "a reducer .* is required",
        ),
    ],
)
def test_eucaim_cdm_dataset_01_bad_arguments(kwargs, match):
    with pytest.raises(FedbiomedValueError, match=match):
        EucaimCdmDataset(**kwargs)


def test_eucaim_cdm_dataset_02_feature_names_without_data():
    dataset = EucaimCdmDataset(
        ["clinical", CT],
        target_modalities="clinical",
        target_columns=[TREATMENT_TYPE],
        **{
            **CLINICAL_ARGS,
            "input_columns": [SEX, START, STUDY_OFFSET],
            "vocabularies": {
                SEX: [fake.MALE, fake.FEMALE],
                TREATMENT_TYPE: [fake.CHEMO, fake.BIO],
            },
        },
    )
    # image_study gives one row (the selected study): no reducer
    assert dataset.input_feature_names == [
        f"{SEX}={fake.MALE}",
        f"{SEX}={fake.FEMALE}",
        f"{START}:mean",
        STUDY_OFFSET,
    ]
    assert dataset.target_feature_names == [
        f"{TREATMENT_TYPE}={fake.CHEMO}",
        f"{TREATMENT_TYPE}={fake.BIO}",
    ]
    assert dataset.data_modalities == {"clinical", CT}
    assert dataset.target_modalities == {"clinical"}


def test_eucaim_cdm_dataset_03_image_only_has_no_feature_names():
    dataset = EucaimCdmDataset(CT)
    assert dataset.input_feature_names is None
    assert dataset.target_feature_names is None
    assert dataset.target_modalities is None


def test_eucaim_cdm_dataset_04_image_study_needs_reducer_without_images():
    with pytest.raises(FedbiomedValueError, match="is required"):
        EucaimCdmDataset("clinical", input_columns=[STUDY_OFFSET])
    dataset = EucaimCdmDataset(
        "clinical", input_columns=[STUDY_OFFSET], reducers={STUDY_OFFSET: "max"}
    )
    assert dataset.input_feature_names == [f"{STUDY_OFFSET}:max"]


def test_eucaim_cdm_dataset_05_overlapping_columns_warn():
    fed_logger = logging.getLogger("fedbiomed")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    fed_logger.addHandler(handler)
    try:
        EucaimCdmDataset(
            "clinical",
            target_modalities="clinical",
            input_columns=[SEX],
            target_columns=[SEX],
        )
    finally:
        fed_logger.removeHandler(handler)
    assert any(
        "both input_columns and target_columns" in r.getMessage() for r in records
    )


def test_eucaim_cdm_dataset_06_not_loaded():
    dataset = EucaimCdmDataset(CT)
    with pytest.raises(FedbiomedError, match="not completed initialization"):
        dataset[0]
    with pytest.raises(FedbiomedError, match="not completed initialization"):
        len(dataset)


# --- samples -----------------------------------------------------------------------------


def test_eucaim_cdm_dataset_07_clinical_only(cdm_root):
    dataset = _loaded(cdm_root, data_modalities="clinical", **CLINICAL_ARGS)
    assert len(dataset) == len(fake.PATIENTS)

    data, target = dataset[0]  # P001: male, treatments at 10 and 20 days
    assert target is None
    assert set(data) == {"clinical"}
    assert data["clinical"].dtype == torch.get_default_dtype()
    np.testing.assert_array_equal(data["clinical"].numpy(), [1, 0, 15])

    data, _ = dataset[fake.PATIENTS.index("P002")]  # female, no treatment
    vector = data["clinical"].numpy()
    np.testing.assert_array_equal(vector[:2], [0, 1])
    assert np.isnan(vector[2])

    data, _ = dataset[fake.PATIENTS.index("P003")]  # days and months mixed
    assert np.isnan(data["clinical"].numpy()[2])


def test_eucaim_cdm_dataset_08_image_only_sklearn(cdm_root):
    dataset = _loaded(cdm_root, DataReturnFormat.SKLEARN, data_modalities=[CT, MR])
    assert len(dataset) == 1
    data, target = dataset[0]
    assert target is None
    assert set(data) == {CT, MR}
    shape = fake.SERIES_OK_SHAPE
    assert data[CT].shape == (shape["cols"], shape["rows"], shape["slices"])
    assert data[CT].dtype == np.float64
    assert data[MR].shape == (3, 3, 2)


@pytest.mark.parametrize("policy, offset", [("earliest", 10), ("latest", 40)])
def test_eucaim_cdm_dataset_09_mixed_with_timepoint_policy(cdm_root, policy, offset):
    dataset = _loaded(
        cdm_root,
        data_modalities=["clinical", CT],
        input_columns=[STUDY_OFFSET],
        timepoint_policy=policy,
    )
    assert len(dataset) == 2  # P001, P003
    data, _ = dataset[1]  # P003: the selected study's offset
    np.testing.assert_array_equal(data["clinical"].numpy(), [offset])
    assert isinstance(data[CT], torch.Tensor)
    assert data[CT].shape == (3, 3, 2)


def test_eucaim_cdm_dataset_10_clinical_target(cdm_root):
    dataset = _loaded(
        cdm_root,
        data_modalities=CT,
        target_modalities="clinical",
        target_columns=[TREATMENT_TYPE],
        vocabularies={TREATMENT_TYPE: [fake.CHEMO, fake.BIO]},
        timepoint_policy="latest",
    )
    targets = [dataset[i][1]["clinical"].numpy().tolist() for i in range(len(dataset))]
    assert targets == [[1, 0], [1, 1]]  # P001: chemo; P003: chemo and biological


# --- transforms --------------------------------------------------------------------------


def test_eucaim_cdm_dataset_11_transform_dict(cdm_root):
    dataset = _loaded(
        cdm_root,
        data_modalities=["clinical", CT],
        input_columns=[SEX],
        transform={"clinical": lambda x: x * 10},
        target_modalities=MR,
        target_transform={MR: lambda x: x[None]},
    )
    data, target = dataset[0]
    np.testing.assert_array_equal(data["clinical"].numpy(), [10, 0])
    assert data[CT].shape == (5, 4, 3)  # identity
    assert target[MR].shape == (1, 3, 3, 2)


def test_eucaim_cdm_dataset_12_transform_callable(cdm_root):
    def transform(data):
        return {k: v + 1 for k, v in data.items()}

    dataset = _loaded(
        cdm_root,
        data_modalities=["clinical", CT],
        input_columns=[SEX],
        transform=transform,
    )
    plain = _loaded(cdm_root, data_modalities=["clinical", CT], input_columns=[SEX])
    data, plain_data = dataset[0][0], plain[0][0]
    for modality in ("clinical", CT):
        torch.testing.assert_close(data[modality], plain_data[modality] + 1)


@pytest.mark.parametrize(
    "transform, match",
    [
        ({"clinical": lambda x: x.tolist()}, "Expected `transform` to return"),
        (lambda data: data["clinical"], "to return a dict"),
        (lambda data: {"clinical": 3}, "Expected transform to return"),
        ({"clinical": lambda x: 1 / 0}, "Unable to apply transform"),
    ],
)
def test_eucaim_cdm_dataset_13_bad_transform_fails_at_load(cdm_root, transform, match):
    dataset = EucaimCdmDataset("clinical", input_columns=[SEX], transform=transform)
    with pytest.raises(FedbiomedError, match=match):
        dataset.load(cdm_root, DataReturnFormat.TORCH)


# --- load errors -------------------------------------------------------------------------


def test_eucaim_cdm_dataset_14_no_sample_left(cdm_root):
    dataset = EucaimCdmDataset("EUCAIM:IMG1000024")
    with pytest.raises(FedbiomedError, match="no sample left"):
        dataset.load(cdm_root, DataReturnFormat.TORCH)


def test_eucaim_cdm_dataset_15_unreadable_first_sample(cdm_root):
    dataset = EucaimCdmDataset(fake.PCCT_CODE)  # P007's series has no folder
    with pytest.raises(FedbiomedError, match=fake.SERIES_NO_DIR):
        dataset.load(cdm_root, DataReturnFormat.TORCH)


def test_eucaim_cdm_dataset_16_missing_root(tmp_path):
    dataset = EucaimCdmDataset(CT)
    with pytest.raises(FedbiomedError, match="Failed to create Controller"):
        dataset.load(tmp_path / "nope", DataReturnFormat.TORCH)


# --- series queries ----------------------------------------------------------------------

T1_QUERY = SeriesQuery(MR, ParameterFilter(fake.TISSUE_CONTRAST, value_code=fake.T1))
T2_QUERY = SeriesQuery(MR, ParameterFilter(fake.TISSUE_CONTRAST, value_code=fake.T2))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"series_queries": [T1_QUERY]}, "must be a dict"),
        ({"series_queries": {"": T1_QUERY}}, "non-empty str"),
        ({"series_queries": {3: T1_QUERY}}, "non-empty str"),
        ({"series_queries": {"clinical": T1_QUERY}}, "is reserved"),
        ({"series_queries": {MR: T1_QUERY}}, "is reserved"),
        ({"series_queries": {"t1": MR}}, "must be a SeriesQuery"),
        (
            {"series_queries": {"t1": SeriesQuery("EUCAIM:IMG1000022")}},
            "not a `series_modality` code",
        ),
        (
            {"series_queries": {"t1": T1_QUERY, "t2": T2_QUERY}},
            r"unselected modalities: \['t2'\]",
        ),
    ],
)
def test_eucaim_cdm_dataset_17_bad_series_queries(kwargs, match):
    with pytest.raises(FedbiomedValueError, match=match):
        EucaimCdmDataset(data_modalities=[CT, "t1"], **kwargs)


def test_eucaim_cdm_dataset_18_unknown_modality_name():
    with pytest.raises(FedbiomedValueError, match="a `series_queries` name"):
        EucaimCdmDataset(data_modalities="t1")


def test_eucaim_cdm_dataset_19_series_queries_property():
    dataset = EucaimCdmDataset(
        data_modalities=["t1", CT],
        target_modalities="t2",
        series_queries={"t1": T1_QUERY, "t2": T2_QUERY},
    )
    assert dataset.series_queries == {
        CT: SeriesQuery(CT),
        "t1": T1_QUERY,
        "t2": T2_QUERY,
    }


def test_eucaim_cdm_dataset_20_named_sequences(cdm_root):
    fake.add_mr_sequences_patient(cdm_root)
    dataset = _loaded(
        cdm_root,
        data_modalities=["clinical", "t1"],
        input_columns=[SEX],
        target_modalities="t2",
        series_queries={"t1": T1_QUERY, "t2": T2_QUERY},
        transform={"t1": lambda x: x[None]},
    )
    assert len(dataset) == 1  # P009 only: P001 has no T2, P004 no T1
    data, target = dataset[0]
    assert set(data) == {"clinical", "t1"}
    np.testing.assert_array_equal(data["clinical"].numpy(), [1, 0])  # male
    assert data["t1"].shape == (1, 3, 3, 2)
    assert isinstance(target["t2"], torch.Tensor)
    assert target["t2"].shape == (3, 3, 2)
