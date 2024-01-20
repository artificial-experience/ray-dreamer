from functools import reduce
from typing import List, Dict, Any, Optional
from ray.tune.registry import register_env
from ray.tune.registry import _global_registry, ENV_CREATOR

from datetime import datetime
from functools import wraps
from os.path import expandvars

import yaml


def load_yaml(yaml_path):
    def process_dict(dict_to_process):
        for key, item in dict_to_process.items():
            if isinstance(item, dict):
                dict_to_process[key] = process_dict(item)
            elif isinstance(item, str):
                dict_to_process[key] = expandvars(item)
            elif isinstance(item, list):
                dict_to_process[key] = process_list(item)
        return dict_to_process

    def process_list(list_to_process):
        new_list = []
        for item in list_to_process:
            if isinstance(item, dict):
                new_list.append(process_dict(item))
            elif isinstance(item, str):
                new_list.append(expandvars(item))
            elif isinstance(item, list):
                new_list.append(process_list(item))
            else:
                new_list.append(item)
        return new_list

    with open(yaml_path) as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    return process_dict(yaml_content)


def get_current_timestamp(use_hour=True):
    if use_hour:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d")


def get_nested_dict_field(
    *, directive: Dict[str, Any], keys: List[str]
) -> Optional[Any]:
    """
    Get a nested value from a dictionary.

    Args:
        directives: The target dictionary.
        keys: A list of keys representing the path to the target value in the dictionary.

    Returns:
        The target value if it exists in the dictionary. Otherwise, returns None.
    """
    return reduce(
        lambda d, key: d.get(key) if isinstance(d, dict) else None, keys, directive
    )


def register_custom_env(env_id, env_creator_func):
    """register a custom environment if it's not already registered"""

    def is_env_registered(env_name):
        is_registered = False
        try:
            # If the environment can be created, then it is registered
            _global_registry.get(ENV_CREATOR, env_name)
            is_registered = True
        except Exception as e:
            pass

        return is_registered

    if not is_env_registered(env_id):
        register_env(env_id, env_creator_func)
        print(f"Environment '{env_id}' is registered.")
    else:
        print(f"Environment '{env_id}' is already registered.")
