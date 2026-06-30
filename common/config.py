from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = REPO_ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
