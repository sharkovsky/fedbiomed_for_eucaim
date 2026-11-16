# EUCAIM CDM Dataset

## Introduction

`EucaimCdmDataset` trains models on cancer imaging datasets that follow the
[EUCAIM](https://eucaim.cancerimage.eu/) Common Data Model (CDM). One sample is one
**patient**. A sample can hold:

- **clinical features**: columns of the CDM clinical tables (patient, cancer condition,
  treatments, imaging studies, ...) encoded into a fixed-length numeric vector;
- **image volumes**: one DICOM series per requested image modality, all from the **same
  imaging study** of the patient.

Every node keeps its data in the same standard model, so the researcher describes the
data with standard concepts (`table.column` names and EUCAIM ontology codes) instead of
site-specific file names. The same training plan runs unchanged on every EUCAIM node.

!!! warning "Preview"
    The EUCAIM CDM dataset is under development in the EUCAIM fork of Fed-BioMed.
    The dataset type (`eucaim-cdm`) is not yet available in `fedbiomed node dataset add`
    or in the GUI, and the classes are not yet exported from `fedbiomed.common.dataset`.
    Import them from their modules, as in the examples below:

    ```python
    from fedbiomed.common.dataset._eucaim_cdm_dataset import EucaimCdmDataset
    from fedbiomed.common.dataset_reader._eucaim_cdm_reader import SeriesQuery
    from eucaim_cdm_reader import ParameterFilter
    ```

## Key Features

- One sample per patient, built from the normalized CDM tables and the patient's DICOM
  series.
- Clinical columns encoded from the CDM schema alone: the feature layout is **identical on
  every node** and **known before training**, without seeing any data.
- Coded columns (EUCAIM ontology codes) become one-hot or multi-hot entries. You can keep
  only the codes you care about.
- Numeric columns of tables with several rows per patient (treatments, procedures, ...)
  are reduced with `min`, `max` or `mean`.
- Image modalities are selected by `series_modality` code (for example *Computed
  tomography*) or by **named queries** on acquisition parameters (for example "MR with
  Tissue Contrast = T1 weighted and Slice Thickness ≤ 3 mm").
- Timepoint selection: all of a sample's volumes come from one study, chosen by a
  `timepoint_policy`.
- PyTorch and scikit-learn formats, with per-modality or global transforms.

## Concepts in Five Minutes

### The dataset on a node

A EUCAIM CDM dataset is a folder with the standard EUCAIM layout:

```
root/
├── clinical_data/
│   ├── patient.csv
│   ├── cancer_condition.csv
│   ├── procedure.csv
│   ├── treatment.csv            (optional)
│   ├── surgical_procedure.csv   (optional)
│   ├── radiotherapy.csv         (optional)
│   ├── episode.csv
│   ├── episode_event.csv
│   ├── image_study.csv
│   ├── image_series.csv
│   └── image_modality.csv       (acquisition parameters, used by series queries)
└── imaging_data/
    └── {patient_id}/{study_uid}/{series_uid}/*.dcm
```

The tables and their columns are defined by the CDM. See the
[EUCAIM CDM browser](https://eucaim-cdm.ics.forth.gr/) for every table, column and allowed
code, and the [EUCAIM hyper-ontology](https://hyperontology.eucaim.cancerimage.eu/) to look
codes up.

Data holders are expected to validate their dataset (for example with
`eucaim-cdm-reader`) before deploying it. Fed-BioMed does **not** re-validate the data. It
only ignores rows that would make the patient/study/series index ambiguous (missing or
duplicated identifiers, studies of unknown patients, ...) and logs a warning on the node.

### Codes

Coded values are EUCAIM ontology codes such as `EUCAIM:IMG1000038` (*Magnetic resonance
imaging*). The codes used in this page:

| Code | Meaning | Where |
|---|---|---|
| `EUCAIM:IMG1000038` | Magnetic resonance imaging | `image_series.series_modality` |
| `EUCAIM:IMG1000042` | Computed tomography | `image_series.series_modality` |
| `EUCAIM:IMG1016649` | Tissue Contrast | acquisition parameter |
| `EUCAIM:IMG1016652` | T1 weighted | Tissue Contrast value |
| `EUCAIM:IMG1016647` | T2 weighted | Tissue Contrast value |
| `EUCAIM:IMG1016306` | Slice Thickness | acquisition parameter |
| `EUCAIM:IMG1016756` | Contrast agent | acquisition parameter |
| `EUCAIM:COM1000152` / `EUCAIM:COM1001954` | millimeter / centimeter | units |
| `EUCAIM:COM1001366` / `EUCAIM:COM1001370` | Male / Female | `patient.patient_birth_sex` |
| `EUCAIM:CLIN1024528` | Chemotherapy | `treatment.treatment_type` |
| `EUCAIM:CLIN1014763` | Biological Therapy | `treatment.treatment_type` |

The allowed codes of any column can be listed without data:

```python
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import EucaimCdmReader

spec = EucaimCdmReader.field_spec("treatment", "treatment_type")
sorted(spec.code_list)   # bare codes, e.g. 'CLIN1014763', 'CLIN1024528', ...
```

### Modalities

Like `MedicalFolderDataset`, a sample is a pair `(data, target)` of dicts keyed by
**modality**. A modality is one of:

| Modality key | Value in the sample |
|---|---|
| `"clinical"` | 1-D vector of the encoded clinical columns |
| a `series_modality` code with its prefix, e.g. `"EUCAIM:IMG1000042"` | volume of the patient's series of that modality |
| a name you define in `series_queries`, e.g. `"t1"` | volume of the patient's series matching that query |

```python
data = {
    "clinical": torch.Tensor([5]),           # encoded clinical columns
    "t1":       torch.Tensor([256, 256, 30]), # T1-weighted MR volume (X, Y, Z)
    "t2":       torch.Tensor([256, 256, 30]), # T2-weighted MR volume, same study
}
target = {"clinical": torch.Tensor([2])}      # or None without target modalities
```

## Quick Start

The dataset is created inside the training plan's `training_data` method. The example
below predicts which treatment types a patient received from their T1- and T2-weighted MR
scans and their sex.

```python
import torch
import torch.nn as nn
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, Resize

from eucaim_cdm_reader import ParameterFilter
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset._eucaim_cdm_dataset import EucaimCdmDataset
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import SeriesQuery
from fedbiomed.common.training_plans import TorchTrainingPlan

MR = "EUCAIM:IMG1000038"
TISSUE_CONTRAST, T1, T2 = "IMG1016649", "IMG1016652", "IMG1016647"
CHEMO, BIO = "EUCAIM:CLIN1024528", "EUCAIM:CLIN1014763"


def make_dataset(transform=None, target_transform=None):
    return EucaimCdmDataset(
        data_modalities=["clinical", "t1", "t2"],
        target_modalities="clinical",
        input_columns=["patient.patient_birth_sex"],
        target_columns=["treatment.treatment_type"],
        vocabularies={"treatment.treatment_type": [CHEMO, BIO]},
        series_queries={
            "t1": SeriesQuery(MR, ParameterFilter(TISSUE_CONTRAST, value_code=T1)),
            "t2": SeriesQuery(MR, ParameterFilter(TISSUE_CONTRAST, value_code=T2)),
        },
        timepoint_policy="earliest",
        transform=transform,
        target_transform=target_transform,
    )


class TreatmentPlan(TorchTrainingPlan):

    def init_dependencies(self):
        return [
            "import torch",
            "from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, Resize",
            "from eucaim_cdm_reader import ParameterFilter",
            "from fedbiomed.common.dataset._eucaim_cdm_dataset import EucaimCdmDataset",
            "from fedbiomed.common.dataset_reader._eucaim_cdm_reader import SeriesQuery",
        ]

    def init_model(self):
        return self.Net(self.model_args())

    class Net(nn.Module):
        ...  # uses model_args["n_clinical"] and model_args["n_targets"]

    def training_data(self):
        image = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize((128, 128, 32)),
            NormalizeIntensity(),
        ])
        dataset = make_dataset(
            transform={
                "t1": image,
                "t2": image,
                "clinical": lambda x: torch.nan_to_num(x),  # missing values are NaN
            },
        )
        return DataManager(dataset, batch_size=2, shuffle=True)

    def training_step(self, data, target):
        logits = self.model()(data["clinical"], data["t1"], data["t2"])
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target["clinical"]
        )
```

Since the feature layout needs no data, the researcher can size the model locally before
starting the experiment:

```python
dataset = make_dataset()
dataset.input_feature_names
# ['patient.patient_birth_sex=EUCAIM:COM1001366',     (the full code list of the
#  'patient.patient_birth_sex=EUCAIM:COM1001370']      column: Male, Female)
dataset.target_feature_names
# ['treatment.treatment_type=EUCAIM:CLIN1024528',
#  'treatment.treatment_type=EUCAIM:CLIN1014763']

model_args = {
    "n_clinical": len(dataset.input_feature_names),
    "n_targets": len(dataset.target_feature_names),
}
```

The constructor checks every argument against the CDM schema. A typo in a column name, an
invalid code or a missing reducer fails **on the researcher's machine**, not on the nodes.

## Clinical Features

### Selecting columns

`input_columns` (and `target_columns`) list `table.column` names, in output order. The
table prefix is needed because column names repeat across tables
(`treatment.start_offset_unit` and `episode.start_offset_unit`, for example).

`"clinical"` must be in `data_modalities` if and only if `input_columns` is given, and in
`target_modalities` if and only if `target_columns` is given.

Supported tables: `patient`, `cancer_condition`, `procedure`, `treatment`,
`surgical_procedure`, `radiotherapy`, `episode`, `image_study`, `image_series`.

Not supported: `episode_event` (it only links other tables), `image_modality` (it is used
to [select series](#selecting-series-by-acquisition-parameters) instead), the segmentation
tables, and the mandatory views.

### How columns are encoded

The CDM fixes how many rows a table can give for one patient. That decides the encoding:

| Table | Rows per sample |
|---|---|
| `patient` | one |
| `image_study` | one when image modalities are requested (the selected study), otherwise many (all the patient's studies) |
| `cancer_condition`, `procedure`, `treatment`, `surgical_procedure`, `radiotherapy`, `episode` | many |
| `image_series` | many (the selected series, or all the patient's series without image modalities) |

| Column kind | One-row table | Many-row table | Feature names |
|---|---|---|---|
| **Coded** (has a code list) | one-hot | multi-hot: 1 if any row has the code | `table.column=EUCAIM:CODE`, one per code |
| **Numeric or boolean** | value passed through | reduced with the reducer you give: `"min"`, `"max"` or `"mean"` | `table.column`, or `table.column:max` when reduced |
| **Date, identifier, free text** | rejected | rejected | n/a |

Booleans are read from `true`/`false`/`1`/`0` (any case) and encoded as 1/0.

```python
EucaimCdmDataset(
    data_modalities="clinical",
    input_columns=[
        "patient.patient_birth_sex",                  # coded, one row: one-hot
        "treatment.treatment_type",                   # coded, many rows: multi-hot
        "treatment.start_offset_from_diagnosis",      # numeric, many rows: reducer needed
        "cancer_condition.cancer_condition_age_at_diagnosis",
    ],
    reducers={
        "treatment.start_offset_from_diagnosis": "max",
        "cancer_condition.cancer_condition_age_at_diagnosis": "min",
    },
)
```

A reducer is **required** for each numeric column of a many-row table, and **rejected**
everywhere else. There are no `first`/`last` reducers, because the CDM tables have no
reliable row order.

### Vocabularies: choosing the codes

By default a coded column gives one entry per code of its full code list, sorted. Many
code lists are long, so you will usually keep only the codes you need:

```python
vocabularies={
    "treatment.treatment_type": ["EUCAIM:CLIN1024528", "CLIN1014763"],  # prefix optional
}
```

- The output has one entry per code, **in the order given**.
- Every code must be in the column's code list, otherwise the constructor raises.
- Values outside the vocabulary, external codes (`ICD10:...`, `SNOMEDCT:...`) and missing
  values all give zeros. No "other" or "missing" entry is added. Add the codes you need to
  the vocabulary if you want to see them.

### Missing values and units

| Situation | Result |
|---|---|
| No row for the patient, or only missing values | `NaN` (numeric), zeros (coded) |
| Optional table absent on a node (e.g. no `radiotherapy.csv`) | as if no row; warning in the node log |
| Optional column absent from a node's CSV | as if no row; warning in the node log |
| Rows to reduce use **different units**, or a unit is missing | `NaN`; warning in the node log |

Units are never converted. For example, if a patient has treatments with offsets in days
and in months, `treatment.start_offset_from_diagnosis:max` is `NaN` for that patient. In a
one-row table the value passes through whatever its unit: add the unit column (for example
`image_study.study_offset_unit`) as a feature if you need it.

!!! tip "Handle NaN in your transform"
    The clinical vector can contain `NaN`. Impute it before the model sees it, for
    example with `transform={"clinical": torch.nan_to_num}` or an imputation of your own.

## Image Modalities

### By `series_modality` code

The simplest image modality is a `series_modality` code, written with its `EUCAIM:`
prefix. It matches any series of that modality:

```python
EucaimCdmDataset(data_modalities=["EUCAIM:IMG1000042"])   # one CT volume per patient
```

### Selecting series by acquisition parameters

A code is not enough when a study holds several series of the same modality: a T1 and a T2
MR are both *Magnetic resonance imaging*. Asking for plain MR would then find two series
in the study and skip it, because nothing tells which one to use.

**Series queries** solve this with the acquisition parameters of `image_modality.csv`.
Define named modalities in `series_queries` and use the names as modality keys:

```python
from eucaim_cdm_reader import ParameterFilter
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import SeriesQuery

MR = "EUCAIM:IMG1000038"

EucaimCdmDataset(
    data_modalities=["t1", "t2"],
    series_queries={
        "t1": SeriesQuery(MR, ParameterFilter("IMG1016649", value_code="IMG1016652")),
        "t2": SeriesQuery(MR, ParameterFilter("IMG1016649", value_code="IMG1016647")),
    },
)
```

A `SeriesQuery` takes:

| Argument | Meaning |
|---|---|
| `modality` | `series_modality` code, or `None` for any modality |
| `parameters` | one `ParameterFilter` or a list of them; a series must satisfy **all** |
| `include_subclasses` | also match ontology subclasses of `modality` and of each `value_code` (see below) |

A `ParameterFilter` is one condition on one acquisition parameter:

```python
ParameterFilter("IMG1016649", value_code="IMG1016652")         # Tissue Contrast = T1 weighted
ParameterFilter("IMG1016306", max_value=3, unit="COM1000152")  # Slice Thickness <= 3 mm
ParameterFilter("IMG1016306", min_value=1, max_value=3, unit="COM1000152")  # 1 to 3 mm
ParameterFilter("IMG1016756")                                   # Contrast agent recorded
```

- Codes are accepted with or without the `EUCAIM:` prefix, in the query and in the data.
- Numeric bounds are inclusive and need a unit. Stored values are converted within the
  same quantity (cm to mm, Gy to mGy). Rows with a missing or incompatible unit never match.
- Parameter codes, value codes and units must be allowed by the CDM schema. The dataset
  constructor checks this, so a wrong code fails on the researcher's machine.

Rules for names:

- A name cannot be `"clinical"` nor start with `EUCAIM:` (those are reserved for the
  clinical vector and for bare codes).
- Every query in `series_queries` must be used in `data_modalities` or
  `target_modalities`.
- One series can satisfy two queries (for example `"mr"` = any MR and `"t1"`). It is then
  returned under both names.

!!! note "Ontology subclasses"
    With `include_subclasses=True`, a query for *Arterial phase* also matches *Early
    arterial phase*, and a query for MR also matches *T1-Weighted Magnetic Resonance
    Imaging* (`IMG1004472`). This needs the EUCAIM ontology, so `rdflib` must be installed
    where the dataset is created: `pip install "eucaim-cdm-reader[ontology]"`. Without it,
    the constructor raises an error that says so.

    The ontology does not nest every related concept: *T1W-CE* is a sibling of *T1
    weighted*, not a subclass. Check the
    [hyper-ontology](https://hyperontology.eucaim.cancerimage.eu/) before relying on
    subclasses.

!!! warning "Nodes without acquisition parameters"
    If a node has no `image_modality.csv`, a query with parameters matches no series on that
    node, and its patients are dropped (a warning is logged on the node). Queries without
    parameters (modality only) still work.

### What a volume looks like

Each image modality gives the patient's series as a 3-D array:

- axis order **(X, Y, Z)**, in **RAS** orientation;
- `RescaleSlope` / `RescaleIntercept` applied (so CT values are in Hounsfield units);
- converted to a `torch.Tensor` (TORCH format) or a `numpy.ndarray` (SKLEARN format), with
  the framework's default float type.

The geometry (affine and voxel spacing) is dropped, as in `MedicalFolderDataset`. Volumes
from different patients or nodes usually have different shapes and spacings: resize or
crop them in a transform (for example MONAI's `Resize`) before batching.

Multi-frame DICOM (including DICOM SEG) cannot be read yet.

## Timepoint Selection

A patient may have several imaging studies (baseline, follow-up, ...). All of a sample's
volumes come from **one** study, so that they describe the same moment.

A study is a **candidate** for a patient when it holds **exactly one** series for each
requested image modality. `timepoint_policy` then picks among the candidates:

| Policy | Behavior |
|---|---|
| `"unique"` (default) | keep the patient only if exactly one study is a candidate |
| `"earliest"` | keep the candidate with the earliest `study_acquisition_date` |
| `"latest"` | keep the candidate with the latest `study_acquisition_date` |

For `"earliest"`/`"latest"`, if any candidate has no acquisition date, the candidates are
ordered by `study_offset_from_diagnosis` instead, but only when they all use the same
`study_offset_unit`. If the candidates still cannot be ordered, or two share the earliest
or latest value, the patient is dropped.

With clinical modalities only, every patient in `patient.csv` is a sample and the policy is
ignored.

`image_study` and `image_series` columns follow the selection: with image modalities,
`image_study` holds the selected study (one row) and `image_series` holds the selected
series.

Each node logs how many patients it dropped, by reason:

| Reason | Meaning |
|---|---|
| `no_imaging` | the patient has no imaging study |
| `missing_modalities` | no study has every requested modality |
| `duplicate_modality` | a study has every modality, but one of them more than once |
| `several_candidates` | policy `"unique"` and more than one candidate study |
| `no_ordering` | policy `"earliest"`/`"latest"` and the candidates cannot be ordered |
| `tie` | policy `"earliest"`/`"latest"` and two candidates share the extreme value |

Patient identifiers are never logged, and these logs stay on the node. If no sample is left
on a node, loading fails there with an error that lists the counts.

## Transformations

Transforms work as in `MedicalFolderDataset`. They are applied after conversion to the
requested format:

- **a dict** of callables keyed by modality: each one gets its own modality's value, and
  modalities without an entry are left unchanged;
- **a single callable**: it gets the whole dict and must return a dict.

```python
transform = {
    "t1": Compose([EnsureChannelFirst(channel_dim="no_channel"), Resize((128, 128, 32))]),
    "clinical": torch.nan_to_num,
}
target_transform = {"clinical": lambda y: y.float()}
```

Transforms are checked on the first sample when the dataset is loaded on the node, so a
broken transform fails early. See [Applying Transformations](applying-transformations.md)
for general guidance.

## Constructor Reference

```python
EucaimCdmDataset(
    data_modalities,                 # str or list: "clinical", codes, series_queries names
    target_modalities=None,          # same, or None for no target
    input_columns=None,              # list of "table.column", required iff "clinical" in data
    target_columns=None,             # same, for the target
    vocabularies=None,               # {"table.column": [codes]}: codes to keep, in order
    reducers=None,                   # {"table.column": "min" | "max" | "mean"}
    timepoint_policy="unique",       # "unique" | "earliest" | "latest"
    transform=None,                  # callable, or dict keyed by data modality
    target_transform=None,           # callable, or dict keyed by target modality
    series_queries=None,             # {"name": SeriesQuery(...)}
)
```

Properties, available without data:

| Property | Value |
|---|---|
| `data_modalities`, `target_modalities` | the requested modalities |
| `input_feature_names`, `target_feature_names` | names of the clinical vector entries, or `None` without `"clinical"` |
| `series_queries` | `{modality: SeriesQuery}` for every image modality; codes appear as `SeriesQuery(code)` |

## Errors

All errors carry the code `FB634`.

**When the dataset is created** (researcher side and node side). These raise
`FedbiomedValueError`:

- unknown modality (neither `"clinical"`, a `series_queries` name, nor a prefixed
  `series_modality` code);
- `input_columns`/`target_columns` missing or given without `"clinical"`;
- unsupported table, unknown column, column that cannot be encoded (date, identifier,
  free text), duplicated column;
- vocabulary code not in the column's code list;
- reducer missing, invalid, or given for a column that does not take one;
- vocabulary, reducer or series query given for something you did not select;
- invalid series query (unknown code or unit, reserved name, `include_subclasses` without
  `rdflib`);
- `target_transform` without `target_modalities`, or transform keys that are not
  modalities.

**When the dataset is loaded on a node:**

- the dataset folder or a mandatory table (`patient`, `image_study`, `image_series`) is
  missing;
- no sample is left after timepoint selection;
- the first sample cannot be read, converted or transformed;
- a selected series cannot be read when a sample is accessed (missing folder, unreadable
  files, slices of different sizes).

## Privacy

- The feature layout depends only on the CDM schema and on your arguments. Nodes reveal
  neither which codes they hold nor how many rows they have.
- Node logs about the data (dropped patients, ignored rows, unreadable series) contain
  local paths and identifiers. They stay on the node and are never sent to the researcher.

## Limitations

- Node registration (`fedbiomed node dataset add`, GUI) is not available yet (see
  [Preview](#introduction)).
- One sample per patient. A study or a series cannot be the sample unit.
- No segmentation targets (DICOM SEG) yet.
- Acquisition parameters (`image_modality`) select series but are not features.
- Units are never converted in clinical reductions: mixed units give `NaN`.
- No federated analytics.
- No data loading plan: modality names are defined by the researcher, through codes and
  series queries.
