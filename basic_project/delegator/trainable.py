from common import methods
from .construct.ppo import PPOConstruct
from .abstract.registration import ConstructRegistry


class TrainableConstructDelegator:
    def __init__(self, construct_directive: dict):
        self.construct_directive = construct_directive

        self._registered_trainable_constructs = None
        self._construct_configuration_file = None
        self._target_trainable_construct = None
        self._target_trainable_ray_prefix = None

    @classmethod
    def from_construct_directive(cls, construct_directive: dict):
        instance = cls(construct_directive)
        instance._construct_configuration_file = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["configuration", "config_name", "choice"],
        )
        instance._target_trainable_construct = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["configuration", "construct_class", "choice"],
        )
        instance._target_trainable_ray_prefix = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["configuration", "ray_prefix", "choice"],
        )
        instance._registered_trainable_constructs = (
            ConstructRegistry.get_registered_constructs()
        )
        return instance

    def delegate(self):
        construct = None
        if self._target_trainable_construct in self._registered_trainable_constructs:
            construct = ConstructRegistry.create(
                construct_type=self._target_trainable_construct,
                path_to_construct_file=self._construct_configuration_file,
            ).commit()
        else:
            raise SystemError(
                f"ERROR: {self._target_trainable_construct} is not registered"
            )
        return construct

    @property
    def target_trainable_ray_prefix(self):
        if self._target_trainable_ray_prefix is not None:
            return self._target_trainable_ray_prefix
        else:
            raise SystemError(f"ERROR: {self._target_trainable_ray_prefix} is None")
