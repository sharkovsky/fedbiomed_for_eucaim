import json
import argparse
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

# Import user-provided training plan
from eucaim_trainin_plan import EucaimTrainingPlan 


# Import user-provided experiment configuration
with open('/app/config/arguments.json', 'r') as f:
    config = json.load(f)

# Unpack configuration
model_args = config["model"]
training_args = config["training"]
training_args["optimizer_args"] = config["optimizer"]
training_args["loader_args"] = config["loader"]
exp_kwargs = config["experiment"]

# Parse runtime arguments 
parser = argparse.ArgumentParser(
        description="Fed-BioMed federated training application for EUCAIM"
    )

parser.add_argument(
    "dataset_id",
    type=str,
    help="The dataset ID (must be alphanumeric, e.g., 'Dataset123')."
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Extract dataset tag from command-line arguments
    tags = [str(args.dataset_id)]

    # Define federated learning experiment
    exp = Experiment(tags=tags,
                     training_plan_class=EucaimTrainingPlan,
                     model_args=model_args,
                     training_args=training_args,
                     aggregator=FedAverage(),
                     node_selection_strategy=None,
                     **exp_kwargs)

    # Run experiment
    exp.run()






