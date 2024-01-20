from ray import air, tune
from common import methods
from .trainable import TrainableConstructDelegator


class TunerDelegator:
    def __init__(self, construct_directive: dict, tuner_directive: dict):
        self.construct_directive = construct_directive
        self.tuner_directive = tuner_directive

        # return _param_space and ray_trainable_prefix
        self._trainable_construct_delegator = None

        self._ray_trainable_prefix = None
        self._param_space = None
        self._ai_runtime_config = None

        # TODO: create this part of a workflow
        self._tune_config = None

    @classmethod
    def from_trial_directive(cls, construct_directive: dict, tuner_directive: dict):
        instance = cls(construct_directive, tuner_directive)
        instance._trainable_construct_delegator = (
            TrainableConstructDelegator.from_construct_directive(
                construct_directive=construct_directive
            )
        )
        instance._setup_run_config(tuner_directive)
        return instance

    def _setup_run_config(self, tuner_directive: dict):
        stop_conditions = {
            "training_iteration": methods.get_nested_dict_field(
                directive=tuner_directive,
                keys=[
                    "ai-runtime-conditions",
                    "stop-config",
                    "training_iteration",
                    "choice",
                ],
            ),
            "timesteps_total": methods.get_nested_dict_field(
                directive=tuner_directive,
                keys=[
                    "ai-runtime-conditions",
                    "stop-config",
                    "timesteps_total",
                    "choice",
                ],
            ),
            "episode_reward_mean": methods.get_nested_dict_field(
                directive=tuner_directive,
                keys=[
                    "ai-runtime-conditions",
                    "stop-config",
                    "episode_reward_mean",
                    "choice",
                ],
            ),
        }

        checkpoint_conditions = air.CheckpointConfig(
            checkpoint_frequency=methods.get_nested_dict_field(
                directive=tuner_directive,
                keys=[
                    "ai-runtime-conditions",
                    "checkpoint-config",
                    "checkpoint_frequency",
                    "choice",
                ],
            ),
            checkpoint_at_end=methods.get_nested_dict_field(
                directive=tuner_directive,
                keys=[
                    "ai-runtime-conditions",
                    "checkpoint-config",
                    "checkpoint_at_end",
                    "choice",
                ],
            ),
        )

        verbose = methods.get_nested_dict_field(
            directive=tuner_directive,
            keys=["ai-runtime-conditions", "run-config", "verbose", "choice"],
        )
        self._ai_runtime_config = air.RunConfig(
            stop=stop_conditions,
            checkpoint_config=checkpoint_conditions,
            verbose=verbose,
        )

    def _setup_trainable_prefix_and_param_space(self):
        self._param_space = self._trainable_construct_delegator.delegate()
        self._ray_trainable_prefix = (
            self._trainable_construct_delegator.target_trainable_ray_prefix
        )

    def delegate_tuner_entity(self):
        """Instantiate and return tuner entity ready for training"""
        self._setup_trainable_prefix_and_param_space()
        tuner = tune.Tuner(
            trainable=self._ray_trainable_prefix,
            param_space=self._param_space,
            run_config=self._ai_runtime_config,
        )

        return tuner
