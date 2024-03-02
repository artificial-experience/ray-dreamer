from src.common import methods, constants
from src import environment
from src.abstract.base_construct import BaseConstruct
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config


class DreamerV3Construct(BaseConstruct):
    def __init__(self, construct_registry_directive: str):
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

    @classmethod
    def from_construct_registry_directive(cls, construct_registry_directive: str):
        instance = cls(construct_registry_directive)
        path_to_construct_file = construct_registry_directive.get(
            "path_to_construct_file", None
        )
        construct_file_path = (
            constants.Directories.TRAINABLE_CONFIG_DIR.value / path_to_construct_file
        )
        instance._construct_configuration = methods.load_yaml(construct_file_path)
        return instance

    def _validate_and_register_custom_env(self):
        custom_env_prefix = self._construct_configuration["env-directive"]["prefix"][
            "choice"
        ]
        custom_env_config = self._construct_configuration["env-directive"][
            "env_config"
        ]["choice"]
        custom_env_class_creator = self._construct_configuration["env-directive"][
            "custom_env_class_creator"
        ]["choice"]

        custom_env_creator_function = getattr(environment, custom_env_class_creator)
        methods.register_custom_env(
            env_id=custom_env_prefix,
            env_creator_func=lambda config: custom_env_creator_function(
                custom_env_config
            ),
        )

    def _env_config(self):
        return {
            "env": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["env-directive", "prefix", "choice"],
            ),
        }

   # def _env_config(self):
   #     return {
   #         "env": "CartPole-v1",
   #     }

    def _framework_config(self):
        return {
            "framework": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "framework", "choice"],
            ),
        }

    def _rollouts_config(self):
        return {
            "num_rollout_workers": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "rollouts", "num-rollout-workers", "choice"],
            ),
            "num_envs_per_worker": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "rollouts", "num-envs-per-worker", "choice"],
            ),
            "rollout_fragment_length": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "rollouts", "rollout_fragment_length", "choice"],
            ),
        }

    def _training_config(self):
        return {
            "model_size": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "training", "model_size", "choice"],
            ),
            "training_ratio": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "training", "training_ratio", "choice"],
            ),
            "lr": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "training", "lr", "choice"],
            ),
            "model": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "training", "model", "choice"],
            ),
        }

    def _resources_config(self):
        return {
            "num_gpus": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "resources", "num_gpus", "choice"],
            ),
        }

    def _evaluation_config(self):
        return {
            "evaluation_interval": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "evaluation", "evaluation-interval", "choice"],
            ),
            "evaluation_duration": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "evaluation", "evaluation-duration", "choice"],
            ),
            "enable_async_evaluation": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["dreamerv3-directive", "evaluation", "enable_async_evaluation", "choice"],
            ),
        }

    def commit(self):
        self._validate_and_register_custom_env()
        construct = DreamerV3Config()

        env = self._env_config()
        construct.environment(**env)

        framework = self._framework_config()
        construct.framework(**framework)

        rollouts = self._rollouts_config()
        construct.rollouts(**rollouts)

        training = self._training_config()
        construct.training(**training)

        resources = self._resources_config()
        construct.resources(**resources)

        evaluation = self._evaluation_config()
        construct.evaluation(**evaluation)

        return construct.to_dict()
