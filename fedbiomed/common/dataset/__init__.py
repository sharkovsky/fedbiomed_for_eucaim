# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset
"""

# Used to build `EucaimCdmDataset(series_queries=...)`, so exported beside it
from fedbiomed.common.dataset_reader import SeriesQuery

from ._custom_dataset import CustomDataset
from ._dataset import Dataset
from ._eucaim_cdm_dataset import EucaimCdmDataset
from ._image_label_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)
from ._mappings import (
    DATASET_CLASSES_PER_TYPE,
    REGISTRY_CONTROLLERS,
    get_controller,
)
from ._medical_folder_dataset import MedicalFolderDataset
from ._tabular_dataset import TabularDataset

__all__ = [
    "Dataset",
    "CustomDataset",
    "EucaimCdmDataset",
    "ImageFolderDataset",
    "MedicalFolderDataset",
    "MedNistDataset",
    "MnistDataset",
    "TabularDataset",
    "SeriesQuery",
    "DATASET_CLASSES_PER_TYPE",
    "REGISTRY_CONTROLLERS",
    "get_controller",
]
