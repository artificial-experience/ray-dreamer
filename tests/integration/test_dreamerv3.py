import ray
from ray import tune
from ray.tune import Tuner
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.air import RunConfig, CheckpointConfig

# Initialize Ray
ray.init()

# Define the configuration for DreamerV3 training
config = (DreamerV3Config()
          .environment("CartPole-v1")
          .training(model_size="XS", training_ratio=1024)
          .resources(num_gpus=0))  # Set num_gpus to 0 since there is no GPU available

# Define the tuner with the desired number of episodes
tuner = Tuner(
    "DreamerV3",
    run_config=RunConfig(
        stop={"training_iteration": 1000},  # Set to 1000 episodes as per the requirement
        checkpoint_config=CheckpointConfig(checkpoint_at_end=True)
    ),
    param_space=config,
)

# Run the tuner
result = tuner.fit()

# Shutdown Ray
ray.shutdown()
