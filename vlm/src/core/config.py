import yaml
from typing import Dict, Any

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}
