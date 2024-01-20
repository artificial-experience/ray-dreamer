from basic_project.common import methods, constants
from basic_project import environment
from delegator.abstract.base_construct import BaseConstruct
from ray.rllib.algorithms.ppo.ppo import PPOConfig


class PPOConstruct(BaseConstruct):
    def __init__(self, construct_registry_directive: dict):
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

    def _framework_config(self):
        return {
            "framework": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "framework", "choice"],
            ),
        }

    def _rollouts_config(self):
        return {
            "num_rollout_workers": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "rollouts", "num-workers", "choice"],
            ),
            "num_envs_per_worker": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "rollouts", "num-envs-per-worker", "choice"],
            ),
            "rollout_fragment_length": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "rollouts", "rollout_fragment_length", "choice"],
            ),
        }

    def _training_config(self):
        return {
            "lr": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "lr", "choice"],
            ),
            "lambda_": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "lambda_", "choice"],
            ),
            "gamma": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "gamma", "choice"],
            ),
            "sgd_minibatch_size": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "sgd_minibatch_size", "choice"],
            ),
            "use_gae": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "use_gae", "choice"],
            ),
            "train_batch_size": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "train_batch_size", "choice"],
            ),
            "num_sgd_iter": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "num_sgd_iter", "choice"],
            ),
            "clip_param": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "clip_param", "choice"],
            ),
            "model": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "training", "model", "choice"],
            ),
        }

    def _resources_config(self):
        return {
            "num_gpus": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "resources", "num_gpus", "choice"],
            ),
        }

    def _evaluation_config(self):
        return {
            "evaluation_interval": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "evaluation", "evaluation-interval", "choice"],
            ),
            "evaluation_duration": methods.get_nested_dict_field(
                directive=self._construct_configuration,
                keys=["ppo-directive", "evaluation", "evaluation-duration", "choice"],
            ),
        }

    def commit(self):
        self._validate_and_register_custom_env()
        construct = PPOConfig()

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
