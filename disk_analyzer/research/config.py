"""Experiment and aggregation configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from sklearn.model_selection import ParameterGrid

from disk_analyzer.research.paths import resolve_path


DEFAULT_TIMES = np.arange(0, 730)
DEFAULT_TRAIN_TIMES_FREQ = 20
DEFAULT_TRAIN_BATCHSIZE = 512


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    early_stopping: bool
    patience: int
    min_delta: float
    score_metric: str
    gradient_accumulation_steps: int
    t_0: int
    t_mult: int
    eta_min: float


@dataclass
class DataConfig:
    data_folder: str
    train_batchsize: int
    score_batchsize: int
    times: np.ndarray
    train_times_freq: int
    to_cens_shift: List[int]
    to_term_shift: List[int]
    cens_prob: float


@dataclass
class ExperimentConfig:
    metrics: List[str]
    schema: Dict[str, list]
    res_filename: str
    models_folder: str
    log_dir: str


@dataclass
class GridSearchConfig:
    exp_num: int
    exp_desc: str
    mode: str  # "train" | "test_only"
    data_folder: str
    data_ext: str
    train_grid: List[int]
    test_grid: List[int]
    val_data_size: int
    metrics: List[str]
    param_grids: Dict[str, ParameterGrid]
    training: TrainingConfig
    data: DataConfig
    files_to_save: List[str]
    base_exp: Optional[int] = None  # for test_only mode
    score_sample_grid: Optional[List[int]] = None
    bootstrap_n: Optional[int] = None
    seed: int = 42
    config_source_path: Optional[str] = None


@dataclass
class AggConfig:
    base_exp_num: int
    dataset_train_samples: int
    model_train_samples: int
    sample_grid: List[int]
    test_samples: List[int]
    model_name: str
    data_folder: str
    data_ext: str
    metrics: List[str]
    aggregator_preset: str
    exp_desc: str
    train_batchsize: int = DEFAULT_TRAIN_BATCHSIZE
    times: Optional[np.ndarray] = None
    config_source_path: Optional[str] = None


@dataclass
class QuantileAggConfig:
    exp_num: int
    base_exp_num: int
    data_folder: str
    data_samples: int
    data_ext: str
    model_name: str
    sample_grid: List[int]
    quantiles: List[int]
    quantile_mode: str
    metrics: List[str]
    aggregator_preset: str
    max_obs_day: int = 1600
    max_time: int = 600
    score_batchsize: int = 512
    seed: int = 42


def _parse_times(raw: Any) -> np.ndarray:
    if isinstance(raw, list) and len(raw) == 2:
        return np.arange(raw[0], raw[1])
    if isinstance(raw, list):
        return np.array(raw)
    raise ValueError(f"Invalid times spec: {raw}")


def _build_param_grids(raw: Dict[str, Dict[str, list]]) -> Dict[str, ParameterGrid]:
    return {name: ParameterGrid(params) for name, params in raw.items()}


def load_grid_search_config(path: str) -> GridSearchConfig:
    config_path = resolve_path(path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data_raw = raw.get("data", {})
    training_raw = raw.get("training", {})
    times = _parse_times(data_raw.get("times", [0, 730]))

    data_cfg = DataConfig(
        data_folder=raw["data_folder"],
        train_batchsize=data_raw.get("train_batchsize", DEFAULT_TRAIN_BATCHSIZE),
        score_batchsize=data_raw.get("score_batchsize", 512),
        times=times,
        train_times_freq=data_raw.get("train_times_freq", DEFAULT_TRAIN_TIMES_FREQ),
        to_cens_shift=data_raw.get("to_cens_shift", []),
        to_term_shift=data_raw.get("to_term_shift", []),
        cens_prob=data_raw.get("cens_prob", -1),
    )

    training_cfg = TrainingConfig(
        epochs=training_raw.get("epochs", 1),
        lr=training_raw.get("lr", 1e-4),
        early_stopping=training_raw.get("early_stopping", True),
        patience=training_raw.get("patience", 30),
        min_delta=training_raw.get("min_delta", 0.001),
        score_metric=training_raw.get("score_metric", "ibs"),
        gradient_accumulation_steps=training_raw.get("gradient_accumulation_steps", 1),
        t_0=training_raw.get("t_0", 2),
        t_mult=training_raw.get("t_mult", 2),
        eta_min=training_raw.get("eta_min", 1e-6),
    )

    param_grids = _build_param_grids(raw.get("param_grids", {}))

    return GridSearchConfig(
        exp_num=raw["exp_num"],
        exp_desc=raw.get("exp_desc", ""),
        mode=raw.get("mode", "train"),
        data_folder=raw["data_folder"],
        data_ext=raw.get("data_ext", "csv"),
        train_grid=raw.get("train_grid", []),
        test_grid=raw.get("test_grid", []),
        val_data_size=raw.get("val_data_size", 20),
        metrics=raw.get("metrics", ["ci", "ibs", "ibs_bal"]),
        param_grids=param_grids,
        training=training_cfg,
        data=data_cfg,
        files_to_save=raw.get("files_to_save", []),
        base_exp=raw.get("base_exp"),
        score_sample_grid=raw.get("score_sample_grid"),
        bootstrap_n=raw.get("bootstrap_n"),
        seed=raw.get("seed", 42),
        config_source_path=str(config_path),
    )


def load_agg_config(path: str) -> AggConfig:
    config_path = resolve_path(path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    times_raw = raw.get("times")
    times = _parse_times(times_raw) if times_raw else None

    return AggConfig(
        base_exp_num=raw["base_exp_num"],
        dataset_train_samples=raw["dataset_train_samples"],
        model_train_samples=raw["model_train_samples"],
        sample_grid=raw["sample_grid"],
        test_samples=raw.get("test_samples", [25]),
        model_name=raw["model_name"],
        data_folder=raw["data_folder"],
        data_ext=raw.get("data_ext", "csv"),
        metrics=raw.get("metrics", ["ci", "ibs", "ibs_bal"]),
        aggregator_preset=raw.get("aggregator_preset", "default"),
        exp_desc=raw.get("exp_desc", ""),
        train_batchsize=raw.get("train_batchsize", DEFAULT_TRAIN_BATCHSIZE),
        times=times,
        config_source_path=str(config_path),
    )


def load_quantile_agg_config(path: str) -> QuantileAggConfig:
    with open(resolve_path(path)) as f:
        raw = yaml.safe_load(f)

    return QuantileAggConfig(
        exp_num=raw["exp_num"],
        base_exp_num=raw["base_exp_num"],
        data_folder=raw["data_folder"],
        data_samples=raw["data_samples"],
        data_ext=raw.get("data_ext", "parquet"),
        model_name=raw["model_name"],
        sample_grid=raw["sample_grid"],
        quantiles=raw["quantiles"],
        quantile_mode=raw.get("quantile_mode", "all"),
        metrics=raw.get("metrics", ["ci", "ibs", "ibs_bal"]),
        aggregator_preset=raw.get("aggregator_preset", "minimal"),
        max_obs_day=raw.get("max_obs_day", 1600),
        max_time=raw.get("max_time", 600),
        score_batchsize=raw.get("score_batchsize", 512),
        seed=raw.get("seed", 42),
    )
