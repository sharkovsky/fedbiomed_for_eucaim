"""Fed-BioMed federated training application for EUCAIM.

Runs a single federated experiment, non-interactively, from inputs provided
by the EUCAIM sandbox:

- the training plan, copied by the entrypoint to /app/custom/eucaim_training_plan.py
- the experiment configuration, copied by the entrypoint to /app/config/arguments.json
- the dataset ID, passed on the command line and used as the dataset tag

Experiment results (including breakpoints) are copied to /sandbox/experiments.

The paths are the container's. Environment variables override them, so that the same
script can run outside the container (see `envs/vpn/docker/eucaim_sandbox/run_local.sh`):
`EUCAIM_ARGUMENTS_FILE`, `EUCAIM_SANDBOX_DIR` (results go to its `experiments`
folder), `FBM_RESEARCHER_COMPONENT_ROOT` and `EUCAIM_NODE_STARTUP_DELAY_S`.
"""

import argparse
import json
import os
import shutil
import time

# Import user-provided training plan
from eucaim_training_plan import EucaimTrainingPlan

from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.federated_workflows import Experiment

CONFIG_FILE = os.environ.get("EUCAIM_ARGUMENTS_FILE", "/app/config/arguments.json")
EXPERIMENTS_DIR = os.path.join(
    os.environ.get("FBM_RESEARCHER_COMPONENT_ROOT", "/fbm-researcher"),
    "var",
    "experiments",
)
RESULTS_DIR = os.path.join(
    os.environ.get("EUCAIM_SANDBOX_DIR", "/sandbox"), "experiments"
)

# Time given to the nodes to start and connect before launching the experiment
NODE_STARTUP_DELAY_S = int(os.environ.get("EUCAIM_NODE_STARTUP_DELAY_S", "30"))


def load_config(path: str) -> tuple[dict, dict, dict]:
    """Load the user-provided experiment configuration.

    Returns:
        model_args, training_args and extra keyword arguments for `Experiment`.
    """
    with open(path, "r") as f:
        config = json.load(f)

    model_args = config["model"]
    training_args = config["training"]
    training_args["optimizer_args"] = config["optimizer"]
    training_args["loader_args"] = config["loader"]
    exp_kwargs = config["experiment"]
    return model_args, training_args, exp_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fed-BioMed federated training application for EUCAIM"
    )
    parser.add_argument(
        "dataset_id",
        type=str,
        help="The dataset ID (must be alphanumeric, e.g., 'Dataset123').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_args, training_args, exp_kwargs = load_config(CONFIG_FILE)

    time.sleep(NODE_STARTUP_DELAY_S)

    # Define federated learning experiment, targeting nodes by dataset tag
    exp = Experiment(
        tags=[str(args.dataset_id)],
        training_plan_class=EucaimTrainingPlan,
        model_args=model_args,
        training_args=training_args,
        aggregator=FedAverage(),
        save_breakpoints=True,
        node_selection_strategy=None,
        **exp_kwargs,
    )

    exp.run()

    # Copy the results to persistent storage
    shutil.copytree(EXPERIMENTS_DIR, RESULTS_DIR, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
