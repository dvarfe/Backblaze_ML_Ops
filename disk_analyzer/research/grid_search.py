"""Grid search orchestration for training and test-only rescoring."""

import copy
import os
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.research.config import (
    DEFAULT_TRAIN_BATCHSIZE,
    ExperimentConfig,
    GridSearchConfig,
)
from disk_analyzer.research.dataloader import prepare_dataloader
from disk_analyzer.research.io import init_experiments_folder, load_model, save_experiment_config
from disk_analyzer.research.paths import artifacts_dir, models_dir, resolve_path
from disk_analyzer.research.runner import run_single_experiment, score_test_dataloader, _write_test_result_rows
from disk_analyzer.research.schema import collect_all_hparams, make_schema, write_dict
from disk_analyzer.stages import ModelScorer


def _load_train_max_df(cfg: GridSearchConfig):
    data_folder = resolve_path(cfg.data_folder)
    max_train = max(cfg.train_grid)
    if cfg.data_ext == 'csv':
        df = pd.read_csv(data_folder / f"{max_train}_train_preprocessed.csv")
    else:
        df = pd.read_parquet(data_folder / f"{max_train}_train_preprocessed.parquet")
    df["duration"] = df["max_lifetime"] - df["time"]
    return df[["duration", "failure", "time"]]


def run_grid_search(cfg: GridSearchConfig) -> None:
    np.random.seed(cfg.seed)

    if cfg.mode == "test_only":
        run_test_only_rescore(cfg)
        return

    if not cfg.param_grids:
        raise ValueError("param_grids must be non-empty for train mode")
    all_hparams = collect_all_hparams(cfg.param_grids)
    schema = make_schema(
        cfg.metrics, all_hparams,
        score_sample_grid=cfg.score_sample_grid,
        bootstrap_n=cfg.bootstrap_n,
    )

    base = artifacts_dir(cfg.exp_num)
    exp_cfg = ExperimentConfig(
        metrics=cfg.metrics,
        schema=schema,
        res_filename=str(base / "grid_search.csv"),
        models_folder=str(models_dir(cfg.exp_num)),
        log_dir=str(base / "logs"),
    )

    _, _, _, _ = init_experiments_folder(
        exp_num=cfg.exp_num,
        exp_cfg=exp_cfg,
        data_cfg=cfg.data,
        train_cfg=cfg.training,
        exp_desc=cfg.exp_desc,
        files_to_save=cfg.files_to_save,
        config_source_path=cfg.config_source_path,
    )

    max_train = max(cfg.train_grid)
    dl_score_max = prepare_dataloader(
        train_samples=max_train,
        data_cfg=cfg.data,
        data_type="train",
        dataset_type="score",
        data_ext=cfg.data_ext,
    )
    df_train_max = _load_train_max_df(cfg)
    scorer = ModelScorer()

    run_id = 0
    for train_samples in cfg.train_grid:
        for method in cfg.param_grids:
            for hparams in cfg.param_grids[method]:
                run_single_experiment(
                    train_samples=train_samples,
                    test_grid=cfg.test_grid,
                    method=method,
                    hparams=hparams,
                    run_id=run_id,
                    scorer=scorer,
                    dl_score_max=dl_score_max,
                    df_train_max=df_train_max,
                    exp_cfg=exp_cfg,
                    data_cfg=cfg.data,
                    train_cfg=cfg.training,
                    val_data_size=cfg.val_data_size,
                    data_ext=cfg.data_ext,
                    score_sample_grid=cfg.score_sample_grid,
                    bootstrap_n=cfg.bootstrap_n,
                    seed=cfg.seed,
                )
                run_id += 1


def run_test_only_rescore(cfg: GridSearchConfig) -> None:
    """Rescore models from a base experiment without retraining."""
    if cfg.base_exp is None:
        raise ValueError("base_exp is required for test_only mode")

    base = artifacts_dir(cfg.exp_num)
    if base.exists():
        raise ValueError(f"Experiment Exp_{cfg.exp_num} already exists")
    base.mkdir(parents=True)

    if cfg.config_source_path:
        save_experiment_config(cfg.exp_num, cfg.config_source_path)

    res_filename = str(base / "eval_grid_search.csv")

    if os.path.exists(res_filename):
        raise ValueError(f"Results file already exists: {res_filename}")

    base_schema = {
        'train_samples': [],
        'test_samples': [],
        'method': [],
        'error': [],
        'error_text': [],
        'model_id': [],
    }
    schema = copy.deepcopy(base_schema)
    for metric in cfg.metrics:
        schema[f'{metric}_train_same_size'] = []
        schema[f'{metric}_train_max_size'] = []
        schema[f'{metric}_test'] = []
    if cfg.score_sample_grid is not None:
        schema['score_samples'] = []
    if cfg.bootstrap_n is not None:
        schema['bootstrap_iter'] = []

    pd.DataFrame(schema).to_csv(res_filename, index=False)

    grid_search = pd.read_csv(artifacts_dir(cfg.base_exp) / "grid_search.csv")
    source_models = models_dir(cfg.base_exp)
    scorer = ModelScorer()
    data_folder = resolve_path(cfg.data_folder)
    train_batchsize = cfg.data.train_batchsize

    max_train = max(cfg.train_grid)
    df_train_max = pd.read_csv(data_folder / f"{max_train}_train_preprocessed.csv")
    df_train_max['duration'] = df_train_max['max_lifetime'] - df_train_max['time']
    df_train_max = df_train_max[['duration', 'failure', 'time']]

    dl_score_max = DataLoader(
        dataset=DiskDataset('score', [str(data_folder / f"{max_train}_train_preprocessed.csv")]),
        batch_size=train_batchsize,
    )

    for _, row in grid_search.iterrows():
        try:
            train_samples = int(row['train_samples'])
            method = row['method']
            model_id = row['model_id']

            model_path = source_models / f"{model_id}_model.pkl"
            if not model_path.exists():
                model_path = source_models / f"{model_id}.pkl"
            model = load_model(str(model_path))

            statistics = copy.deepcopy(schema)
            statistics['train_samples'] = [train_samples]
            statistics['method'] = [method]
            statistics['error'] = [0]
            statistics['error_text'] = ['']
            statistics['model_id'] = [model_id]

            for col in row.index:
                if col in statistics and col not in ('train_samples', 'method', 'model_id'):
                    statistics[col] = [row[col]]

            dl_train_score = DataLoader(
                dataset=DiskDataset('score', [str(data_folder / f"{train_samples}_train_preprocessed.csv")]),
                batch_size=train_batchsize,
            )
            df_train_predictions, df_train_gt = model.predict(dl_train_score, cfg.data.times)
            train_metrics = scorer.get_metrics(
                model, df_train_predictions, df_train_gt, cfg.data.times,
                metrics=cfg.metrics,
                df_train=df_train_max,
            )
            for metric in cfg.metrics:
                statistics[f'{metric}_train_same_size'] = [train_metrics.get(metric)]

            df_train_max_predictions, df_train_max_gt = model.predict(dl_score_max, cfg.data.times)
            train_max_metrics = scorer.get_metrics(
                model, df_train_max_predictions, df_train_max_gt, cfg.data.times,
                metrics=cfg.metrics,
                df_train=df_train_max,
            )
            for metric in cfg.metrics:
                statistics[f'{metric}_train_max_size'] = [train_max_metrics.get(metric)]

            multi_test_rows = cfg.score_sample_grid is not None or cfg.bootstrap_n is not None

            for test_samples in cfg.test_grid:
                dl_test_score = DataLoader(
                    dataset=DiskDataset(
                        'score',
                        [str(data_folder / f"{train_samples}_{test_samples}_test_preprocessed.csv")],
                    ),
                    batch_size=train_batchsize,
                )
                try:
                    test_results = score_test_dataloader(
                        model, method, dl_test_score, cfg.data.times, scorer,
                        cfg.metrics, df_train_max, cfg.score_sample_grid,
                        bootstrap_n=cfg.bootstrap_n, seed=cfg.seed,
                    )
                    _write_test_result_rows(
                        statistics, test_results, test_samples, cfg.metrics,
                        res_filename, multi_test_rows,
                    )
                except Exception as e:
                    cur_statistics = copy.deepcopy(statistics)
                    cur_statistics['test_samples'] = [test_samples]
                    cur_statistics['error'] = [1]
                    cur_statistics['error_text'] = [f'EVAL_TEST_ERROR${str(e)}']
                    write_dict(res_filename, cur_statistics)
            print(f"Done with model: {model_id}")
        except Exception as e:
            print(f"Error with model {row.get('model_id', '?')}: {e}")
