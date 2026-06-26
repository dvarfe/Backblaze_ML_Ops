"""Single-experiment training and scoring."""

import copy
import os
import time
from typing import Any, Dict, List, Optional

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from disk_analyzer.models.Cox import (
    CoxTimeInvariantLNFitter,
    CoxTimeInvariantSNFitter,
    CoxTimeVaryingEstimator,
)
from disk_analyzer.models.sksurv_estimators import (
    GradientBoostingSurvivalEstimator,
    RandomSurvivalForestEstimator,
)
from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.models.DLClassifier import DLClassifier
from disk_analyzer.models.DynamicDeepHit import DDH
from disk_analyzer.models.SurvPredictor import SurvPredictor
from disk_analyzer.research.config import DataConfig, ExperimentConfig, TrainingConfig
from disk_analyzer.research.dataloader import prepare_dataloader
from disk_analyzer.research.io import save_model
from disk_analyzer.research.observation_sampling import align_gt_to_predictions, select_nth_observation
from disk_analyzer.research.schema import write_dict
from disk_analyzer.stages import ModelScorer


def prepare_model(method: str, hparams: Dict[str, Any], epochs: int, lr: float):
    if method == "NN":
        return DLClassifier(29, hidden_dim=hparams["hidden_dim"], epochs=epochs, lr=lr)
    if method == "SP":
        return SurvPredictor(29, hidden_dim=hparams["hidden_dim"], epochs=epochs, lr=lr)
    if method == "DDH":
        return DDH(**hparams)
    if method == "CoxTV":
        return CoxTimeVaryingEstimator(**hparams)
    if method == "CoxTISN":
        return CoxTimeInvariantSNFitter(**hparams)
    if method == "CoxTILN":
        return CoxTimeInvariantLNFitter(**hparams)
    if method == "RSF":
        return RandomSurvivalForestEstimator(**hparams)
    if method == "GBSA":
        return GradientBoostingSurvivalEstimator(**hparams)
    raise ValueError(f"Unknown method {method}")


def fit_model(
    model,
    method: str,
    dl_train,
    dl_val,
    train_cfg: TrainingConfig,
    data_cfg: DataConfig,
    writer,
) -> None:
    if method in {"NN"}:
        model.fit(dl_train, writer)

    elif method == "SP":
        optimizer = model.optimizer
        CosineAnnealingWarmRestarts(optimizer, train_cfg.t_0, train_cfg.t_mult, train_cfg.eta_min)
        model.fit(
            dl_train,
            times=data_cfg.times,
            val_dataloader=dl_val,
            early_stopping=train_cfg.early_stopping,
            score_metric=train_cfg.score_metric,
            patience=train_cfg.patience,
            min_delta=train_cfg.min_delta,
            writer=writer,
        )

    elif method == "DDH":
        model.fit(dl_train, times=data_cfg.times, val_dataloader=dl_val)
        print(f'Число параметров: {model.count_parameters()}')

    elif method.startswith("Cox") or method in {"RSF", "GBSA"}:
        model.fit(dl_train)


def _compute_test_metric_results(
    scorer: ModelScorer,
    model,
    preds,
    gt,
    times,
    metrics: List[str],
    df_train_max,
    bootstrap_n: Optional[int],
    seed: int,
) -> List[Dict[str, float]]:
    if bootstrap_n:
        return scorer.bootstrap_metrics(
            model, preds, gt, times, set(metrics),
            n_bootstrap=bootstrap_n, seed=seed, df_train=df_train_max,
        )
    return [scorer.get_metrics(
        model, preds, gt, times,
        metrics=metrics,
        df_train=df_train_max,
    )]


def score_test_dataloader(
    model,
    method: str,
    dl_test,
    times,
    scorer: ModelScorer,
    metrics: List[str],
    df_train_max,
    score_sample_grid: Optional[List[int]] = None,
    bootstrap_n: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Score a test dataloader, optionally filtering to the n-th observation per disk.

    Returns a list of dicts with keys:
        score_samples (int or None), bootstrap_iter (int or None), test_time, metrics.
    """
    results: List[Dict[str, Any]] = []

    if score_sample_grid and method == "DDH":
        trunc_right = max(score_sample_grid)
        for n in score_sample_grid:
            t0 = time.time()
            preds, gt = model.predict(
                dl_test, times, agg_horizon=n, trunc_right=trunc_right,
            )
            metric_results = _compute_test_metric_results(
                scorer, model, preds, gt, times, metrics, df_train_max,
                bootstrap_n, seed,
            )
            elapsed = time.time() - t0
            for bootstrap_iter, test_metrics in enumerate(metric_results):
                results.append({
                    'score_samples': n,
                    'bootstrap_iter': bootstrap_iter if bootstrap_n else None,
                    'test_time': elapsed,
                    'metrics': test_metrics,
                })
    elif score_sample_grid:
        t0 = time.time()
        preds, gt = model.predict(dl_test, times)
        preds = preds.sort_values(['serial_number', 'time'])
        gt = gt.sort_values(['serial_number', 'time'])
        predict_time = time.time() - t0
        for n in score_sample_grid:
            cur_pred = select_nth_observation(preds, n)
            cur_gt = align_gt_to_predictions(cur_pred, gt)
            metric_results = _compute_test_metric_results(
                scorer, model, cur_pred, cur_gt, times, metrics, df_train_max,
                bootstrap_n, seed,
            )
            for bootstrap_iter, test_metrics in enumerate(metric_results):
                results.append({
                    'score_samples': n,
                    'bootstrap_iter': bootstrap_iter if bootstrap_n else None,
                    'test_time': predict_time,
                    'metrics': test_metrics,
                })
    else:
        t0 = time.time()
        preds, gt = model.predict(dl_test, times)
        metric_results = _compute_test_metric_results(
            scorer, model, preds, gt, times, metrics, df_train_max,
            bootstrap_n, seed,
        )
        elapsed = time.time() - t0
        for bootstrap_iter, test_metrics in enumerate(metric_results):
            results.append({
                'score_samples': None,
                'bootstrap_iter': bootstrap_iter if bootstrap_n else None,
                'test_time': elapsed,
                'metrics': test_metrics,
            })

    return results


def _write_test_result_rows(
    base_stats: Dict[str, list],
    test_results: List[Dict[str, Any]],
    test_samples: int,
    metrics: List[str],
    res_filename: str,
    multi_row: bool,
) -> None:
    if multi_row:
        for tr in test_results:
            row = copy.deepcopy(base_stats)
            row["test_samples"] = [test_samples]
            if tr["score_samples"] is not None:
                row["score_samples"] = [tr["score_samples"]]
            if tr["bootstrap_iter"] is not None:
                row["bootstrap_iter"] = [tr["bootstrap_iter"]]
            row["test_time"] = [tr["test_time"]]
            for m in metrics:
                row[f"{m}_test"] = [tr["metrics"][m]]
            write_dict(res_filename, row)
    else:
        tr = test_results[0]
        base_stats["test_time"] = [tr["test_time"]]
        for m in metrics:
            base_stats[f"{m}_test"] = [tr["metrics"][m]]
        write_dict(res_filename, base_stats)


def run_single_experiment(
    train_samples: int,
    test_grid: List[int],
    method: str,
    hparams: Dict[str, Any],
    run_id: int,
    scorer: ModelScorer,
    dl_score_max,
    df_train_max,
    exp_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    val_data_size: int,
    data_ext: str,
    score_sample_grid: Optional[List[int]] = None,
    bootstrap_n: Optional[int] = None,
    seed: int = 42,
):
    stats = copy.deepcopy(exp_cfg.schema)
    stats["train_samples"] = [train_samples]
    stats["method"] = [method]
    stats["model_id"] = [f"{run_id}_{method}"]
    stats["error"] = [0]
    stats["error_text"] = [""]

    for k, v in hparams.items():
        stats[k] = [v]

    writer = SummaryWriter(os.path.join(exp_cfg.log_dir, f"{method}_{run_id}"))

    try:
        dl_train = prepare_dataloader(train_samples, data_cfg, "train", "train", data_ext=data_ext)
        dl_train_score = prepare_dataloader(train_samples, data_cfg, "train", "score", data_ext=data_ext)
        dl_val = prepare_dataloader(
            train_samples, data_cfg, "train", "score", val_data_size, data_ext=data_ext
        )

        model = prepare_model(method, hparams, train_cfg.epochs, train_cfg.lr)

        t0 = time.time()
        fit_model(model, method, dl_train, dl_val, train_cfg, data_cfg, writer)
        stats["train_time"] = [time.time() - t0]

        preds, gt = model.predict(dl_train_score, data_cfg.times)
        train_metrics = scorer.get_metrics(
            model, preds, gt, data_cfg.times,
            metrics=exp_cfg.metrics,
            df_train=df_train_max,
        )
        for m in exp_cfg.metrics:
            stats[f"{m}_train_same_size"] = [train_metrics[m]]

        preds, gt = model.predict(dl_score_max, data_cfg.times)
        max_metrics = scorer.get_metrics(
            model, preds, gt, data_cfg.times,
            metrics=exp_cfg.metrics,
            df_train=df_train_max,
        )
        for m in exp_cfg.metrics:
            stats[f"{m}_train_max_size"] = [max_metrics[m]]

        save_model(model, stats["model_id"][0], exp_cfg.models_folder)

        multi_test_rows = score_sample_grid is not None or bootstrap_n is not None
        base_stats = copy.deepcopy(stats) if multi_test_rows else stats

        for ts in test_grid:
            dl_test = prepare_dataloader(
                train_samples, data_cfg, "test", "score", ts, data_ext=data_ext,
            )
            test_results = score_test_dataloader(
                model, method, dl_test, data_cfg.times, scorer,
                exp_cfg.metrics, df_train_max, score_sample_grid,
                bootstrap_n=bootstrap_n, seed=seed,
            )
            _write_test_result_rows(
                base_stats, test_results, ts, exp_cfg.metrics,
                exp_cfg.res_filename, multi_test_rows,
            )

    except Exception as e:
        stats["error"] = [1]
        stats["error_text"] = [str(e)]
        write_dict(exp_cfg.res_filename, stats)

    return stats
