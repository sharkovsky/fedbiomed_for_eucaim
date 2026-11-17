"""Registration of the `eucaim-cdm` dataset type: registry, exports, node CLI and
dataset manager, and the reload done by the node before training."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from testsupport import fake_eucaim_cdm as fake

import fedbiomed.common.dataset as dataset_pkg
import fedbiomed.common.dataset_controller as controller_pkg
import fedbiomed.common.dataset_reader as reader_pkg
from fedbiomed.common.constants import DatasetTypes, TrainingPlans
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import _mappings
from fedbiomed.common.dataset._eucaim_cdm_dataset import EucaimCdmDataset
from fedbiomed.common.dataset_controller._eucaim_cdm_controller import (
    EucaimCdmController,
)
from fedbiomed.common.dataset_reader._eucaim_cdm_reader import (
    EucaimCdmReader,
    SeriesQuery,
)
from fedbiomed.common.db import TinyDBConnector
from fedbiomed.node.cli_utils import _database as cli_database
from fedbiomed.node.cli_utils._io import DATA_TYPES, validated_data_type_input
from fedbiomed.node.dataset_manager import DatasetManager


@pytest.fixture
def cdm_root(tmp_path: Path) -> Path:
    return fake.build_cdm_dataset(tmp_path / "cdm")


# --- registry and exports ----------------------------------------------------------------


def test_eucaim_cdm_registration_01_registry_entries():
    assert _mappings.DATASET_CLASSES_PER_TYPE[DatasetTypes.EUCAIM_CDM] is (
        EucaimCdmDataset
    )
    assert _mappings.REGISTRY_CONTROLLERS[DatasetTypes.EUCAIM_CDM] == (
        EucaimCdmController,
        EucaimCdmDataset,
    )


def test_eucaim_cdm_registration_02_get_controller(cdm_root):
    controller = _mappings.get_controller("eucaim-cdm", {"root": str(cdm_root)})
    assert isinstance(controller, EucaimCdmController)
    assert controller._controller_kwargs == {"root": str(cdm_root)}


def test_eucaim_cdm_registration_03_package_exports():
    assert dataset_pkg.EucaimCdmDataset is EucaimCdmDataset
    assert dataset_pkg.SeriesQuery is SeriesQuery
    assert controller_pkg.EucaimCdmController is EucaimCdmController
    assert reader_pkg.EucaimCdmReader is EucaimCdmReader
    assert reader_pkg.SeriesQuery is SeriesQuery
    for pkg, names in (
        (dataset_pkg, ("EucaimCdmDataset", "SeriesQuery")),
        (controller_pkg, ("EucaimCdmController",)),
        (reader_pkg, ("EucaimCdmReader", "SeriesQuery")),
    ):
        assert set(names) <= set(pkg.__all__)


# --- node CLI ----------------------------------------------------------------------------


def test_eucaim_cdm_registration_04_cli_menu_appends_type():
    # Existing menu numbers are unchanged; the new type comes last
    assert DATA_TYPES == (
        "csv",
        "default",
        "mednist",
        "images",
        "medical-folder",
        "custom",
        "eucaim-cdm",
    )
    with patch("builtins.input", return_value="7"):
        assert validated_data_type_input() == "eucaim-cdm"


def test_eucaim_cdm_registration_05_cli_add_from_file_keeps_type(cdm_root):
    # `dataset add --file` calls this non-interactively. An unknown type would be
    # silently turned into "default" (MNIST).
    manager = MagicMock()
    manager.list_my_datasets.return_value = []
    cli_database.add_database(
        manager,
        interactive=False,
        path=str(cdm_root),
        name="EUCAIM demo",
        tags="ds123",
        description="EUCAIM CDM dataset",
        data_type="eucaim-cdm",
    )
    manager.add_database.assert_called_once()
    kwargs = manager.add_database.call_args.kwargs
    assert kwargs["data_type"] == "eucaim-cdm"
    assert kwargs["path"] == str(cdm_root)
    assert kwargs["tags"] == ["ds123"]


# --- dataset manager and reload before training ------------------------------------------


@pytest.fixture
def registered(tmp_path, cdm_root, monkeypatch):
    # The TinyDB connector is a process-wide singleton that ignores later paths: give
    # this test its own database, and restore the previous connector afterwards.
    monkeypatch.setattr(TinyDBConnector, "_instance", None)
    manager = DatasetManager(path=str(tmp_path / "db.json"))
    dataset_id = manager.add_database(
        name="EUCAIM demo",
        data_type="eucaim-cdm",
        tags=["ds123"],
        description="EUCAIM CDM dataset",
        path=str(cdm_root),
    )
    entry, _ = manager.get_dataset_entry_by_id(dataset_id)
    return entry


def test_eucaim_cdm_registration_06_dataset_manager_entry(registered, cdm_root):
    controller = EucaimCdmController(cdm_root)
    assert registered["data_type"] == "eucaim-cdm"
    assert registered["path"] == str(cdm_root)
    assert registered["dataset_parameters"] == {}
    assert registered["shape"] == controller.shape()
    assert registered["dtypes"] == controller.get_types()


def test_eucaim_cdm_registration_07_reload_as_node_round(registered):
    dataset = EucaimCdmDataset(
        data_modalities="clinical", input_columns=["patient.patient_birth_sex"]
    )
    data_manager = DataManager(dataset)
    data_manager.load(tp_type=TrainingPlans.TorchTrainingPlan)
    # As `fedbiomed.node.round.Round` builds the controller arguments
    data_manager.load_dataset(
        **{**registered.get("dataset_parameters", {}), "root": registered["path"]}
    )

    assert len(dataset) == registered["shape"]["patients"]
    data, target = dataset[0]
    assert tuple(data["clinical"].shape) == (len(dataset.input_feature_names),)
    assert target is None
