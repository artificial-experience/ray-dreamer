from ray import air, tune
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config

def initialize_config():
    config = (
        DreamerV3Config()
        .environment("Pendulum-v1")
        .training(model_size="XS", training_ratio=1024)
    )
    return config


if __name__ == "__main__":
    trainable_config = initialize_config()
    tuner = tune.Tuner(
        trainable="DreamerV3",
        param_space=trainable_config,
    )
    tuner.fit()
