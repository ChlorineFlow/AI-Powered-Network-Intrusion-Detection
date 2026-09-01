"""Load and expose the ml/config.yaml settings as a plain dict."""

from pathlib import Path
import yaml

# ml/config.yaml lives two levels above this file (ml/src/utils/ -> ml/)
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load the YAML config file and return it as a dict.

    Raises FileNotFoundError if the config file is missing, and
    yaml.YAMLError if it is malformed — both are deliberately left
    unhandled so pipeline startup fails loudly rather than silently
    falling back to defaults.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import json
    print(json.dumps(load_config(), indent=2))