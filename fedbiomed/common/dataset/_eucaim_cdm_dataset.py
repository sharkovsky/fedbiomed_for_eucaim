# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset for EUCAIM CDM datasets: clinical columns and DICOM series, one sample per
patient.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import torch
from eucaim_cdm_reader import DicomVolume

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._eucaim_cdm_controller import (
    TIMEPOINT_POLICIES,
    EucaimCdmController,
)
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import (
    EucaimCdmReader,
    SeriesQuery,
)
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetDataItem,
    Transform,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError
from fedbiomed.common.logger import logger

from ._dataset import Dataset
from ._eucaim_cdm_encoding import EUCAIM_PREFIX, EncodingPlan, parse_columns

CLINICAL = "clinical"


def _err(msg: str) -> FedbiomedValueError:
    """Builds a `FedbiomedValueError` with the EUCAIM CDM error code (FB634)."""
    return FedbiomedValueError(f"{ErrorNumbers.FB634.value}: {msg}")


def _to_numpy(x: Any) -> np.ndarray:
    """Converts a modality value (clinical vector or volume) to a float ndarray.

    The volume's affine and spacing are dropped.
    """
    if isinstance(x, DicomVolume):
        return np.asarray(x.array, dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x
    raise FedbiomedError(
        f"{ErrorNumbers.FB634.value}: unsupported type for format conversion: "
        f"{type(x).__name__}"
    )


class EucaimCdmDataset(Dataset):
    """Dataset over a EUCAIM CDM dataset, one sample per patient.

    A sample is `(data, target)`, each a dict keyed by modality:
    - `"clinical"`: the requested clinical columns, encoded into a 1-D float vector
      (see `input_feature_names` / `target_feature_names`);
    - an image modality: the volume of the patient's series of that modality. It is
      either a name defined in `series_queries` (for example `"t1"`: MR series whose
      Tissue Contrast is T1 weighted), or a `series_modality` code (for example
      `"EUCAIM:IMG1000042"`, any CT series). All of a sample's volumes come from one
      study, chosen with `timepoint_policy`.
    `target` is `None` if no target modality is given.

    Example: T1 and T2 MR volumes of the same study, with a clinical target::

        from eucaim_cdm_reader import ParameterFilter

        TISSUE_CONTRAST, T1, T2 = "IMG1016649", "IMG1016652", "IMG1016647"
        EucaimCdmDataset(
            data_modalities=["t1", "t2"],
            target_modalities="clinical",
            target_columns=["patient.patient_birth_sex"],
            series_queries={
                "t1": SeriesQuery(
                    "EUCAIM:IMG1000038", ParameterFilter(TISSUE_CONTRAST, value_code=T1)
                ),
                "t2": SeriesQuery(
                    "EUCAIM:IMG1000038", ParameterFilter(TISSUE_CONTRAST, value_code=T2)
                ),
            },
        )
    """

    _controller_cls = EucaimCdmController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: _to_numpy,
        DataReturnFormat.TORCH: lambda x: torch.from_numpy(_to_numpy(x)),
    }

    def __init__(
        self,
        data_modalities: Union[str, Iterable[str]],
        target_modalities: Optional[Union[str, Iterable[str]]] = None,
        input_columns: Optional[Iterable[str]] = None,
        target_columns: Optional[Iterable[str]] = None,
        vocabularies: Optional[Mapping[str, Iterable[str]]] = None,
        reducers: Optional[Mapping[str, str]] = None,
        timepoint_policy: str = "unique",
        transform: Transform = None,
        target_transform: Transform = None,
        series_queries: Optional[Mapping[str, SeriesQuery]] = None,
    ):
        """Checks the arguments. No data is needed.

        Args:
            data_modalities: `"clinical"`, names defined in `series_queries`, and/or
                `series_modality` codes with their `EUCAIM:` prefix (shorthand for
                `SeriesQuery(code)`).
            target_modalities: Same, for the target. `None` for no target.
            input_columns: `table.column` names encoded into the `"clinical"` data
                vector. Required if and only if `"clinical"` is in `data_modalities`.
            target_columns: Same for the target.
            vocabularies: `{table.column: codes}` to encode only some codes of a coded
                column (with or without prefix, in output order). Default: the
                column's full code list, sorted.
            reducers: `{table.column: "min" | "max" | "mean"}`, required for each
                numeric column of a table that can give several rows per sample.
            timepoint_policy: How to pick the study of a patient's images:
                `"unique"`, `"earliest"` or `"latest"` (see
                `EucaimCdmController.select`).
            transform: Callable applied to the whole data dict, or dict of callables
                keyed by data modality.
            target_transform: Same, for the target.
            series_queries: `{name: SeriesQuery}` defining named image modalities by
                `series_modality` and acquisition parameters (`image_modality.csv`).
                A study provides a named modality when exactly one of its series
                matches the query. Names cannot be `"clinical"` nor start with
                `EUCAIM:`, and each must be used in `data_modalities` or
                `target_modalities`.

        Raises:
            FedbiomedValueError: if an argument is invalid.
        """
        series_queries = self._validate_series_queries(series_queries)
        self._data_modalities = self._normalize_modalities(
            data_modalities, "data_modalities", series_queries
        )
        self._target_modalities = (
            None
            if target_modalities is None
            else self._normalize_modalities(
                target_modalities, "target_modalities", series_queries
            )
        )
        if timepoint_policy not in TIMEPOINT_POLICIES:
            raise _err(
                f"`timepoint_policy` must be one of {list(TIMEPOINT_POLICIES)}, got "
                f"{timepoint_policy!r}"
            )
        self._timepoint_policy = timepoint_policy

        all_modalities = self._data_modalities | (self._target_modalities or set())
        unused = sorted(set(series_queries) - all_modalities)
        if unused:
            raise _err(f"`series_queries` given for unselected modalities: {unused}")
        # {modality: query} of every image modality; codes become `SeriesQuery(code)`
        self._series_queries: Dict[str, SeriesQuery] = {
            m: series_queries[m] if m in series_queries else SeriesQuery(m)
            for m in sorted(all_modalities - {CLINICAL})
        }
        with_images = bool(self._series_queries)

        self._input_plan = self._make_plan(
            CLINICAL in self._data_modalities,
            input_columns,
            "input_columns",
            "data_modalities",
            vocabularies,
            reducers,
            with_images,
        )
        self._target_plan = self._make_plan(
            CLINICAL in (self._target_modalities or set()),
            target_columns,
            "target_columns",
            "target_modalities",
            vocabularies,
            reducers,
            with_images,
        )
        self._check_unused_entries(
            input_columns if self._input_plan else None,
            target_columns if self._target_plan else None,
            vocabularies,
            reducers,
        )

        self._transform = self._validate_transform(transform, self._data_modalities)
        if self._target_modalities is None:
            if target_transform is not None:
                raise _err(
                    "`target_transform` provided but `target_modalities` is None"
                )
            self._target_transform = None
        else:
            self._target_transform = self._validate_transform(
                target_transform, self._target_modalities
            )

    # === Properties ===
    @property
    def data_modalities(self) -> set:
        """Data modalities: `"clinical"`, `series_queries` names and/or
        `series_modality` codes."""
        return set(self._data_modalities)

    @property
    def target_modalities(self) -> Optional[set]:
        """Target modalities, or `None`."""
        return None if self._target_modalities is None else set(self._target_modalities)

    @property
    def series_queries(self) -> Dict[str, SeriesQuery]:
        """Query of each image modality, codes included (as `SeriesQuery(code)`)."""
        return dict(self._series_queries)

    @property
    def input_feature_names(self) -> Optional[List[str]]:
        """Names of the entries of the `"clinical"` data vector, or `None`."""
        return None if self._input_plan is None else self._input_plan.feature_names()

    @property
    def target_feature_names(self) -> Optional[List[str]]:
        """Names of the entries of the `"clinical"` target vector, or `None`."""
        return None if self._target_plan is None else self._target_plan.feature_names()

    # === Argument checks ===
    @staticmethod
    def _validate_series_queries(
        series_queries: Optional[Mapping[str, SeriesQuery]],
    ) -> Dict[str, SeriesQuery]:
        """Checks the names and queries of `series_queries`; returns them as a dict."""
        if series_queries is None:
            return {}
        if not isinstance(series_queries, Mapping):
            raise _err("`series_queries` must be a dict of {name: SeriesQuery}")
        for name, query in series_queries.items():
            if not (isinstance(name, str) and name):
                raise _err(
                    f"`series_queries` names must be non-empty str, got {name!r}"
                )
            if name == CLINICAL or name.startswith(EUCAIM_PREFIX):
                raise _err(
                    f"`series_queries` name {name!r} is reserved: names cannot be "
                    f"'{CLINICAL}' nor start with '{EUCAIM_PREFIX}'"
                )
            if not isinstance(query, SeriesQuery):
                raise _err(
                    f"`series_queries[{name!r}]` must be a SeriesQuery, got "
                    f"{type(query).__name__}"
                )
            EucaimCdmReader.check_series_query(query)
        return dict(series_queries)

    @staticmethod
    def _normalize_modalities(
        modalities: Union[str, Iterable[str]],
        arg: str,
        series_queries: Mapping[str, SeriesQuery],
    ) -> set:
        """Checks modalities (`"clinical"`, `series_queries` names or
        `series_modality` codes) and returns them as a set."""
        if isinstance(modalities, str):
            modalities = [modalities]
        if isinstance(modalities, dict) or not isinstance(modalities, Iterable):
            raise _err(f"`{arg}` must be a str or an iterable of str")
        modalities = list(modalities)
        if not modalities:
            raise _err(f"`{arg}` cannot be empty")

        code_list = EucaimCdmReader.field_spec(
            "image_series", "series_modality"
        ).code_list
        for modality in modalities:
            if not isinstance(modality, str):
                raise _err(f"`{arg}` must only contain str, got {modality!r}")
            if modality == CLINICAL or modality in series_queries:
                continue
            if not (
                modality.startswith(EUCAIM_PREFIX)
                and modality[len(EUCAIM_PREFIX) :] in code_list
            ):
                raise _err(
                    f"modality {modality!r} in `{arg}` is neither '{CLINICAL}', a "
                    "`series_queries` name, nor a `series_modality` code with its "
                    "'EUCAIM:' prefix"
                )
        return set(modalities)

    @staticmethod
    def _make_plan(
        has_clinical: bool,
        columns: Optional[Iterable[str]],
        columns_arg: str,
        modalities_arg: str,
        vocabularies: Optional[Mapping[str, Iterable[str]]],
        reducers: Optional[Mapping[str, str]],
        with_images: bool,
    ) -> Optional[EncodingPlan]:
        """Builds the encoding plan of `columns`, given iff `"clinical"` is requested."""
        if not has_clinical:
            if columns is not None:
                raise _err(
                    f"`{columns_arg}` given but '{CLINICAL}' is not in `{modalities_arg}`"
                )
            return None
        if columns is None:
            raise _err(
                f"'{CLINICAL}' is in `{modalities_arg}`: `{columns_arg}` is required"
            )
        return EncodingPlan(columns, vocabularies, reducers, with_images)

    @staticmethod
    def _check_unused_entries(
        input_columns: Optional[Iterable[str]],
        target_columns: Optional[Iterable[str]],
        vocabularies: Optional[Mapping[str, Iterable[str]]],
        reducers: Optional[Mapping[str, str]],
    ) -> None:
        """Rejects vocabularies / reducers of unselected columns; warns when a column is
        both an input and a target."""
        inputs = (
            set()
            if input_columns is None
            else set(parse_columns(input_columns, "input_columns"))
        )
        targets = (
            set()
            if target_columns is None
            else set(parse_columns(target_columns, "target_columns"))
        )
        for arg, entries in (("vocabularies", vocabularies), ("reducers", reducers)):
            unused = sorted(set(entries or {}) - inputs - targets)
            if unused:
                raise _err(f"`{arg}` given for unselected column(s): {unused}")
        overlap = sorted(inputs & targets)
        if overlap:
            logger.warning(
                f"Columns {overlap} are present in both input_columns and target_columns."
            )

    @staticmethod
    def _validate_transform(
        transform: Transform, modalities: set
    ) -> Union[Callable, Dict[str, Callable]]:
        """Checks a transform: a callable, or a dict of callables keyed by modality
        (missing modalities get the identity)."""
        transform = {} if transform is None else transform
        if isinstance(transform, dict):
            if not set(transform).issubset(modalities):
                raise _err(
                    "unexpected keys in transform: expected keys among the modalities, "
                    f"got {list(transform)}"
                )
            if not all(callable(v) for v in transform.values()):
                raise _err("transform dict must map modalities to callables")
            return {m: transform.get(m, lambda x: x) for m in modalities}
        if callable(transform):
            return transform
        raise _err("unexpected type for transform: expected a callable or a dict")

    # === Loading ===
    def load(self, root: Union[str, Path], to_format: DataReturnFormat) -> None:
        """Opens the dataset, selects the samples and checks the transforms on the
        first sample.

        Args:
            root: Dataset root folder.
            to_format: Format of the returned data.

        Raises:
            FedbiomedError: if the dataset cannot be opened, no sample is left after
                selection, or the format conversion or a transform fails on sample 0.
        """
        self.to_format = to_format
        self._init_controller(root=root)

        tables = set()
        for plan in (self._input_plan, self._target_plan):
            if plan is not None:
                tables.update(plan.tables)
        self._controller.select(self._series_queries, self._timepoint_policy, tables)

        data, target = self._raw_items(0)
        self._validate_format_and_transformations(data, self._transform)
        if target is not None:
            self._validate_format_and_transformations(
                target, self._target_transform, is_target=True
            )

    def _raw_items(self, idx: int) -> tuple:
        """Returns the data and target dicts of sample `idx`, before conversion."""
        sample = self._controller.get_sample(idx)

        def items(modalities, plan):
            return {
                m: plan.encode(sample["clinical"]) if m == CLINICAL else sample[m]
                for m in modalities
            }

        data = items(self._data_modalities, self._input_plan)
        target = (
            None
            if self._target_modalities is None
            else items(self._target_modalities, self._target_plan)
        )
        return data, target

    # Adapted from `MedicalFolderDataset`
    def _validate_format_and_transformations(
        self,
        data: Dict[str, Any],
        transform: Transform,
        is_target: bool = False,
    ) -> None:
        """Checks the format conversion and the transform on one sample.

        Raises:
            FedbiomedError: if the conversion or the transform fails, or the transform
                does not return the expected type.
        """
        for modality in data:
            data[modality] = self._validate_format_conversion(
                data[modality], extra_info=f"Modality: {modality}"
            )

        if isinstance(transform, dict):
            for modality in data:
                self._validate_transformation(
                    data=data[modality],
                    transform=transform[modality],
                    extra_info=f"Error raised by modality: '{modality}'",
                )
            return

        try:
            data = transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: Failed to apply "
                f"{'target_transform' if is_target else 'transform'} to entire data dict"
            ) from e
        if not isinstance(data, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: Expected "
                f"{'target_transform' if is_target else 'transform'} to return a dict, "
                f"got {type(data).__name__}"
            )
        for modality, value in data.items():
            if not isinstance(value, self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB634.value}: Expected transform to return "
                    f"`{self._to_format.value}` for modality '{modality}', got "
                    f"{type(value).__name__}"
                )

    # === Items ===
    # Adapted from `MedicalFolderDataset`
    def _process_sample_data(
        self,
        data: Dict[str, Any],
        transform: Union[Callable, Dict[str, Callable]],
        idx: int,
        is_target: bool = False,
    ) -> Dict[str, Any]:
        """Converts each modality to the return format, then applies the transform and
        the default types."""
        transform_type = "target_transform" if is_target else "transform"
        to_default = self._get_default_types_callable()
        convert = self._get_format_conversion_callable()
        data = {m: to_default(convert(v)) for m, v in data.items()}

        if isinstance(transform, dict):
            for modality in data:
                try:
                    data[modality] = to_default(transform[modality](data[modality]))
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB634.value}: Failed to apply `{transform_type}` "
                        f"to modality '{modality}' from sample (index={idx}) in "
                        f"{self._to_format.value} format."
                    ) from e
            return data

        try:
            data = transform(data)
            return {m: to_default(v) for m, v in data.items()}
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: Failed to apply `{transform_type}` to "
                f"sample (index={idx}) in {self._to_format.value} format."
            ) from e

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        """Returns `(data, target)` of sample `idx`, converted and transformed."""
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: Dataset object has not completed "
                "initialization. It is not ready to use yet."
            )
        data, target = self._raw_items(idx)
        data = self._process_sample_data(data, self._transform, idx)
        if target is not None:
            target = self._process_sample_data(
                target, self._target_transform, idx, is_target=True
            )
        return data, target
