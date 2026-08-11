import os
from types import SimpleNamespace

import yaml


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "config", "default.yaml")


def load_config(path=None):
    config_path = path or os.environ.get("ECONOMICGRASP_CONFIG", DEFAULT_CONFIG_PATH)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return SimpleNamespace(**data)


cfgs = load_config()
