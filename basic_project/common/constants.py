import os
from enum import Enum
from pathlib import Path


class Directories(Enum):
    RLROOT_DIR = Path(os.getenv("RLROOT", "."))
    CORE_DIR = RLROOT_DIR / "basic_project"
    CONFIG_DIR = CORE_DIR / "config"
    TRAINABLE_CONFIG_DIR = CONFIG_DIR / "trainable"
    CUSTOM_ENV_DIR = CORE_DIR / "environment"
