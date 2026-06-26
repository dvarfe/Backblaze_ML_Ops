"""Repository-root-relative paths for experiments and artifacts."""

from pathlib import Path
from typing import Union


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_CONFIG_NAME = "experiment_config.yaml"
AGGREGATION_CONFIG_NAME = "aggregation_config.yaml"
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs" / EXPERIMENT_CONFIG_NAME
DEFAULT_AGGREGATION_CONFIG = REPO_ROOT / "configs" / AGGREGATION_CONFIG_NAME


def artifacts_dir(exp_num: int) -> Path:
    return REPO_ROOT / "Artifacts" / f"Exp_{exp_num}"


def models_dir(exp_num: int) -> Path:
    return artifacts_dir(exp_num) / "models"


def aggregation_dir(exp_num: int) -> Path:
    return artifacts_dir(exp_num) / "Aggregation"


def logs_dir(exp_num: int) -> Path:
    return artifacts_dir(exp_num) / "logs"


def code_dir(exp_num: int) -> Path:
    return artifacts_dir(exp_num) / "code"


def grid_search_csv(exp_num: int) -> Path:
    return artifacts_dir(exp_num) / "grid_search.csv"


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative paths from repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
