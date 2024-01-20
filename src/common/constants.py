import os
from enum import Enum
from pathlib import Path


class Directories(Enum):
    RLROOT_DIR = Path(os.getenv("RLROOT", "."))
    CORE_DIR = RLROOT_DIR / "src"
    CONFIG_DIR = CORE_DIR / "conf"
    TRAINABLE_CONFIG_DIR = CONFIG_DIR / "trainable"
    CUSTOM_ENV_DIR = CORE_DIR / "environment"
