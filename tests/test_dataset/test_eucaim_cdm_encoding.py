import logging

import numpy as np
import pandas as pd
import pytest
from eucaim_cdm_reader import schema
from eucaim_cdm_reader.schema import BOOLEAN, CODE, INTEGER, NUMBER
from testsupport import fake_eucaim_cdm as fake

from fedbiomed.common.dataset._eucaim_cdm_encoding import (
    UNIT_PAIRS,
    EncodingPlan,
    is_one_row_table,
)
from fedbiomed.common.dataset_controller._eucaim_cdm_controller import CLINICAL_TABLES
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import EucaimCdmReader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

MALE, FEMALE = "EUCAIM:COM1001366", "EUCAIM:COM1001370"  # patient_birth_sex
CHEMO, BIO = "EUCAIM:CLIN1024528", "EUCAIM:CLIN1014763"  # treatment_type
DAY, MONTH = "EUCAIM:COM1000153", "EUCAIM:COM1000154"

SEX = "patient.patient_birth_sex"
TREATMENT_TYPE = "treatment.treatment_type"
START = "treatment.start_offset_from_diagnosis"


def _df(**columns):
    """DataFrame with nullable string columns, or Int64 for int lists."""
    out = {}
    for name, values in columns.items():
        dtype = "Int64" if any(isinstance(v, int) for v in values) else "string"
        out[name] = pd.array(values, dtype=dtype)
    return pd.DataFrame(out)


def _treatments(starts, units, types=None):
    return {
        "treatment": _df(
            start_offset_from_diagnosis=starts,
            start_offset_unit=units,
            treatment_type=types or [CHEMO] * len(starts),
        )
    }


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def warnings_logged():
    """Messages of the warnings reaching the Fed-BioMed logger's handlers."""
    fed_logger = logging.getLogger("fedbiomed")
    previous_level = fed_logger.level
    fed_logger.setLevel(logging.WARNING)
    handler = _ListHandler()
    fed_logger.addHandler(handler)
    yield lambda: [r.getMessage() for r in handler.records]
    fed_logger.removeHandler(handler)
    fed_logger.setLevel(previous_level)


# --- layout ------------------------------------------------------------------------------


def test_eucaim_cdm_encoding_01_rows_per_sample():
    assert is_one_row_table("patient", with_images=False)
    assert is_one_row_table("image_study", with_images=True)
    assert not is_one_row_table("image_study", with_images=False)
    for table in set(CLINICAL_TABLES) - {"patient", "image_study"}:
        assert not is_one_row_table(table, with_images=True)


def test_eucaim_cdm_encoding_02_default_vocabulary_is_full_sorted_code_list():
    plan = EncodingPlan([SEX])
    assert plan.feature_names() == [f"{SEX}={MALE}", f"{SEX}={FEMALE}"]

    code_list = EucaimCdmReader.field_spec("treatment", "treatment_type").codes
    names = EncodingPlan([TREATMENT_TYPE]).feature_names()
    assert names == [f"{TREATMENT_TYPE}=EUCAIM:{c}" for c in sorted(code_list)]


def test_eucaim_cdm_encoding_03_feature_names_and_tables():
    plan = EncodingPlan(
        [START, SEX, TREATMENT_TYPE, "image_study.study_offset_from_diagnosis"],
        vocabularies={TREATMENT_TYPE: [BIO]},
        reducers={START: "max"},
        with_images=True,
    )
    assert plan.feature_names() == [
        f"{START}:max",
        f"{SEX}={MALE}",
        f"{SEX}={FEMALE}",
        f"{TREATMENT_TYPE}={BIO}",
        "image_study.study_offset_from_diagnosis",
    ]
    assert plan.tables == ("treatment", "patient", "image_study")


def test_eucaim_cdm_encoding_04_width_matches_feature_names_for_every_column():
    """Every encodable column of every supported table, in one plan."""
    columns, reducers = [], {}
    for table in CLINICAL_TABLES:
        for spec in schema.TABLES[table].fields:
            name = f"{table}.{spec.name}"
            numeric = spec.kind in (BOOLEAN, INTEGER, NUMBER)
            if spec.kind == CODE or numeric:
                columns.append(name)
                if numeric and not is_one_row_table(table, with_images=False):
                    reducers[name] = "mean"
    plan = EncodingPlan(columns, reducers=reducers)

    vector = plan.encode({})  # no table at all on this node
    assert vector.shape == (len(plan.feature_names()),)
    assert vector.dtype == np.float64


def test_eucaim_cdm_encoding_05_unit_pairs_match_the_schema():
    for value, unit in UNIT_PAIRS.items():
        table, column = value.split(".", 1)
        unit_table, unit_column = unit.split(".", 1)
        assert table == unit_table
        spec = EucaimCdmReader.field_spec(table, column)
        assert spec.kind in (INTEGER, NUMBER)
        assert EucaimCdmReader.field_spec(table, unit_column).kind == CODE


def test_eucaim_cdm_encoding_06_layout_does_not_depend_on_local_data():
    plan_a = EncodingPlan([SEX, TREATMENT_TYPE, START], reducers={START: "min"})
    plan_b = EncodingPlan([SEX, TREATMENT_TYPE, START], reducers={START: "min"})
    node_a = {"patient": _df(patient_birth_sex=[MALE]), **_treatments([3], [DAY])}
    node_b = {
        "patient": _df(patient_birth_sex=[FEMALE]),
        **_treatments([5, 7], [DAY, DAY], [BIO, BIO]),
    }
    assert plan_a.feature_names() == plan_b.feature_names()
    assert plan_a.encode(node_a).shape == plan_b.encode(node_b).shape


# --- coded columns -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (MALE, [1, 0]),
        ("COM1001370", [0, 1]),  # bare code
        ("SNOMEDCT:248153007", [0, 0]),  # external code
        ("ICD10:COM1001366", [0, 0]),  # external prefix never matches EUCAIM
        ("EUCAIM:COM9999999", [0, 0]),  # outside the vocabulary
        (pd.NA, [0, 0]),
    ],
)
def test_eucaim_cdm_encoding_07_one_hot(value, expected):
    plan = EncodingPlan([SEX])
    rows = {"patient": _df(patient_birth_sex=[value])}
    np.testing.assert_array_equal(plan.encode(rows), expected)


def test_eucaim_cdm_encoding_08_multi_hot_in_vocabulary_order():
    plan = EncodingPlan([TREATMENT_TYPE], vocabularies={TREATMENT_TYPE: [BIO, CHEMO]})
    rows = _treatments([1, 2, 3], [DAY] * 3, [CHEMO, CHEMO, "ICD10:X"])
    np.testing.assert_array_equal(plan.encode(rows), [0, 1])
    rows = _treatments([1, 2], [DAY] * 2, [CHEMO, BIO])
    np.testing.assert_array_equal(plan.encode(rows), [1, 1])
    rows = _treatments([], [])
    np.testing.assert_array_equal(plan.encode(rows), [0, 0])


def test_eucaim_cdm_encoding_09_vocabulary_forms():
    # With or without prefix, duplicates removed keeping the first occurrence
    plan = EncodingPlan(
        [TREATMENT_TYPE], vocabularies={TREATMENT_TYPE: ["CLIN1014763", CHEMO, BIO]}
    )
    assert plan.feature_names() == [
        f"{TREATMENT_TYPE}={BIO}",
        f"{TREATMENT_TYPE}={CHEMO}",
    ]


@pytest.mark.parametrize(
    "vocabulary, match",
    [
        (["EUCAIM:CLIN0000000"], "not in the code list"),
        (["ICD10:CLIN1024528"], "not in the code list"),
        ([3], "not in the code list"),
        ([], "is empty"),
    ],
)
def test_eucaim_cdm_encoding_10_bad_vocabulary(vocabulary, match):
    with pytest.raises(FedbiomedValueError, match=match):
        EncodingPlan([TREATMENT_TYPE], vocabularies={TREATMENT_TYPE: vocabulary})


def test_eucaim_cdm_encoding_11_vocabulary_for_numeric_column():
    with pytest.raises(FedbiomedValueError, match="not coded"):
        EncodingPlan([START], vocabularies={START: [CHEMO]}, reducers={START: "min"})


# --- numeric columns ---------------------------------------------------------------------


@pytest.mark.parametrize("reducer, expected", [("min", 3), ("max", 9), ("mean", 6)])
def test_eucaim_cdm_encoding_12_reducers(reducer, expected):
    plan = EncodingPlan([START], reducers={START: reducer})
    rows = _treatments([9, 3, 6, pd.NA], [DAY, DAY, DAY, MONTH])
    np.testing.assert_array_equal(plan.encode(rows), [expected])


def test_eucaim_cdm_encoding_13_float_column():
    name = "cancer_condition.cancer_condition_age_at_diagnosis"
    plan = EncodingPlan([name], reducers={name: "max"})
    rows = {
        "cancer_condition": pd.DataFrame(
            {
                "cancer_condition_age_at_diagnosis": pd.array(
                    [55.5, 61.25], dtype="Float64"
                ),
                "cancer_condition_age_unit": pd.array(["EUCAIM:U"] * 2, dtype="string"),
            }
        )
    }
    np.testing.assert_array_equal(plan.encode(rows), [61.25])


def test_eucaim_cdm_encoding_14_one_row_value_passes_through():
    name = "image_study.study_offset_from_diagnosis"
    plan = EncodingPlan([name], with_images=True)
    rows = {"image_study": _df(study_offset_from_diagnosis=[30])}
    np.testing.assert_array_equal(plan.encode(rows), [30])


@pytest.mark.parametrize(
    "starts, units",
    [
        ([], []),  # no rows
        ([pd.NA, pd.NA], [DAY, DAY]),  # only missing values
    ],
)
def test_eucaim_cdm_encoding_15_no_value_gives_nan(starts, units):
    plan = EncodingPlan([START], reducers={START: "mean"})
    assert np.isnan(plan.encode(_treatments(starts, units))).all()


@pytest.mark.parametrize(
    "units",
    [
        [DAY, MONTH],  # different units
        [DAY, pd.NA],  # unknown unit
    ],
)
def test_eucaim_cdm_encoding_16_mixed_units_give_nan_and_warn_once(
    units, warnings_logged
):
    plan = EncodingPlan([START], reducers={START: "min"})
    rows = _treatments([1, 2], units)
    assert np.isnan(plan.encode(rows)).all()
    assert np.isnan(plan.encode(rows)).all()
    messages = [m for m in warnings_logged() if "units" in m]
    assert len(messages) == 1
    assert START in messages[0]


def test_eucaim_cdm_encoding_17_units_need_checking_only_across_values():
    plan = EncodingPlan([START], reducers={START: "min"})
    # A single value: no mixing, its unit does not matter
    np.testing.assert_array_equal(plan.encode(_treatments([4], [pd.NA])), [4])
    # Missing values do not count
    rows = _treatments([4, pd.NA], [DAY, MONTH])
    np.testing.assert_array_equal(plan.encode(rows), [4])


def test_eucaim_cdm_encoding_18_absent_unit_column():
    plan = EncodingPlan([START], reducers={START: "max"})
    rows = {"treatment": _df(start_offset_from_diagnosis=[1, 2])}
    assert np.isnan(plan.encode(rows)).all()


@pytest.mark.parametrize("value, expected", [(True, 1), (False, 0), (pd.NA, np.nan)])
def test_eucaim_cdm_encoding_19_boolean(value, expected):
    """The reader types boolean columns with the nullable `boolean` dtype."""
    plan = EncodingPlan(["patient.patient_deceased"])
    rows = {"patient": pd.DataFrame({"patient_deceased": pd.array([value], "boolean")})}
    np.testing.assert_array_equal(plan.encode(rows), [expected])


def test_eucaim_cdm_encoding_20_boolean_literals_read_from_csv(tmp_path):
    """Boolean literals of the CSV, as typed by the reader, then encoded.

    The reader accepts true/false/1/0 in any case and reads anything else as missing.
    """
    literals = ["true", " TRUE ", "1", "False", "0", "", "maybe"]
    patients = (
        "patient_id,patient_birth_sex,patient_managing_organization_id,"
        "patient_diagnostic_category,patient_deceased\n"
    ) + "".join(
        f"P{i},{MALE},ORG-1,EUCAIM:COM1001087,{value}\n"
        for i, value in enumerate(literals)
    )
    root = fake.build_cdm_dataset(tmp_path / "cdm", overrides={"patient": patients})
    table = EucaimCdmReader(root).table("patient")

    plan = EncodingPlan(["patient.patient_deceased"])
    encoded = [plan.encode({"patient": table.iloc[[i]]})[0] for i in range(len(table))]
    np.testing.assert_array_equal(encoded, [1, 1, 1, 0, 0, np.nan, np.nan])


# --- missing data ------------------------------------------------------------------------


def test_eucaim_cdm_encoding_21_missing_table_or_column(warnings_logged):
    plan = EncodingPlan([SEX, START], reducers={START: "max"})
    for rows in (
        {},  # table absent from the rows
        {"patient": pd.DataFrame(), "treatment": pd.DataFrame()},  # optional table
        {  # columns absent from the CSV
            "patient": _df(patient_id=["P1"]),
            "treatment": _df(treatment_id=["T1"]),
        },
    ):
        vector = plan.encode(rows)
        np.testing.assert_array_equal(vector[:2], [0, 0])
        assert np.isnan(vector[2])
    # Once per column
    messages = [m for m in warnings_logged() if "absent on this node" in m]
    assert len(messages) == 2


def test_eucaim_cdm_encoding_22_one_row_table_with_several_rows():
    plan = EncodingPlan([SEX])
    with pytest.raises(FedbiomedError, match="one row per sample"):
        plan.encode({"patient": _df(patient_birth_sex=[MALE, FEMALE])})


# --- argument checks ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "columns, reducers, with_images, match",
    [
        ([START], {}, False, "reducer .* is required"),
        (["image_study.study_offset_from_diagnosis"], {}, False, "is required"),
        (
            ["image_study.study_offset_from_diagnosis"],
            {"image_study.study_offset_from_diagnosis": "max"},
            True,
            "one row per sample",
        ),
        ([START], {START: "median"}, False, "must be one of"),
        ([TREATMENT_TYPE], {TREATMENT_TYPE: "max"}, False, "coded column"),
    ],
)
def test_eucaim_cdm_encoding_23_reducer_checks(columns, reducers, with_images, match):
    with pytest.raises(FedbiomedValueError, match=match):
        EncodingPlan(columns, reducers=reducers, with_images=with_images)


@pytest.mark.parametrize(
    "columns, error, match",
    [
        (["patient.patient_birth_date"], FedbiomedValueError, "is a date"),
        (["patient.patient_id"], FedbiomedValueError, "identifier or free text"),
        (["image_series.series_description"], FedbiomedValueError, "free text"),
        (["episode_event.episode_id"], FedbiomedValueError, "unsupported table"),
        (["patient"], FedbiomedValueError, "'table.column'"),
        ([SEX, SEX], FedbiomedValueError, "duplicated"),
        ([], FedbiomedValueError, "cannot be empty"),
        (3, FedbiomedValueError, "iterable"),
        (["patient.nope"], FedbiomedError, "has no column 'nope'"),
    ],
)
def test_eucaim_cdm_encoding_24_column_checks(columns, error, match):
    with pytest.raises(error, match=match):
        EncodingPlan(columns)
