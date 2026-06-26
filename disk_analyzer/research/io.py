"""Experiment folder initialization and model I/O."""

import os
import pickle
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from disk_analyzer.research.config import DataConfig, ExperimentConfig, TrainingConfig
from disk_analyzer.research.paths import (
    AGGREGATION_CONFIG_NAME,
    EXPERIMENT_CONFIG_NAME,
    aggregation_dir,
    artifacts_dir,
    code_dir,
    logs_dir,
    models_dir,
    resolve_path,
)


def save_model(model, model_id: str, folder: str) -> None:
    with open(os.path.join(folder, f"{model_id}.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_config(f, name: str, cfg) -> None:
    f.write(f"[{name}]\n")
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    for k, v in cfg.items():
        f.write(f"{k} = {v}\n")
    f.write("\n")


def save_experiment_config(exp_num: int, config_source_path: str) -> None:
    """Copy the YAML used to launch the run into Artifacts/Exp_N/experiment_config.yaml."""
    src = resolve_path(config_source_path)
    if not src.exists():
        raise FileNotFoundError(f"Experiment config not found: {src}")
    dst = artifacts_dir(exp_num) / EXPERIMENT_CONFIG_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def save_aggregation_config(base_exp_num: int, config_source_path: str, model_name: str) -> None:
    """Copy aggregation YAML into Aggregation/ with a model_name postfix."""
    src = resolve_path(config_source_path)
    if not src.exists():
        raise FileNotFoundError(f"Aggregation config not found: {src}")
    stem = Path(AGGREGATION_CONFIG_NAME).stem
    dst = aggregation_dir(base_exp_num) / f"{stem}_{model_name}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def init_experiments_folder(
    exp_num: int,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    exp_desc: str,
    files_to_save: List[str],
    config_source_path: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    base = artifacts_dir(exp_num)
    models = models_dir(exp_num)
    logs = logs_dir(exp_num)
    code = code_dir(exp_num)
    res_file = base / "grid_search.csv"
    desc_file = base / "Description.txt"

    if base.exists():
        raise ValueError(f"Experiment Exp_{exp_num} already exists")

    models.mkdir(parents=True)
    logs.mkdir(parents=True)
    code.mkdir(parents=True)

    pd.DataFrame(exp_cfg.schema).to_csv(res_file, index=False)

    with open(desc_file, "w") as f:
        dump_config(f, "Experiment", {
            "exp_num": exp_num,
            "metrics": exp_cfg.metrics,
        })
        dump_config(f, "DataConfig", data_cfg)
        dump_config(f, "TrainingConfig", train_cfg)
        f.write("[Description]\n")
        f.write(exp_desc)

    for file in files_to_save:
        src = resolve_path(file)
        if src.exists():
            shutil.copy(src, code)

    if config_source_path:
        save_experiment_config(exp_num, config_source_path)

    return str(base), str(models), str(logs), str(res_file)
