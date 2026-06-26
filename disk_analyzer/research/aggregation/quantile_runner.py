"""Quantile-based aggregation evaluation (time-sliced histories)."""

import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from disk_analyzer.models.Dataset import DataFrameDataset
from disk_analyzer.research.aggregation.presets import get_aggregators
from disk_analyzer.research.config import QuantileAggConfig
from disk_analyzer.research.io import load_model
from disk_analyzer.research.paths import artifacts_dir, models_dir, resolve_path
from disk_analyzer.stages.model_scoring import ModelScorer


def get_quantile_time_points(
    df: pd.DataFrame,
    quantiles: List[int],
    mode: str = "all",
    max_obs_day: int = 1600,
) -> np.ndarray:
    if mode == "equal_events":
        event_times = df.loc[
            (df["failure"] == 1) & (df["max_lifetime"] <= max_obs_day),
            "max_lifetime",
        ].sort_values().values
        total_events = len(event_times)
        if total_events == 0:
            raise ValueError("No failure events found in data")
        time_points = []
        for q in quantiles:
            idx = int(total_events * q / 100)
            if idx >= total_events:
                idx = total_events - 1
            time_points.append(event_times[idx])
        return np.array(time_points)

    if mode == "failures_only":
        event_times = df.loc[
            (df["failure"] == 1) & (df["max_lifetime"] <= max_obs_day),
            "max_lifetime",
        ]
        return np.percentile(event_times, quantiles)

    event_times = df.loc[df["max_lifetime"] <= max_obs_day, "max_lifetime"]
    return np.percentile(event_times, quantiles)


def get_extended_times(df, agg_dict, times):
    any_dict_key = list(agg_dict.keys())[0]
    any_dict_key_key = list(agg_dict[any_dict_key].keys())[0]
    return agg_dict[any_dict_key][any_dict_key_key].get_extended_times(df, times)


def build_history_at_time(
    df: pd.DataFrame,
    t_point: float,
    sample_grid: List[int],
    max_time: int,
) -> Dict[int, pd.DataFrame]:
    window_end = t_point + max_time
    df_alive = df[df["max_lifetime"] > t_point].copy()
    df_in_window = df_alive[df_alive["max_lifetime"] <= window_end]
    df_in_window = df_in_window[df_in_window["time"] <= t_point]
    df_in_window = df_in_window.sort_values(["serial_number", "time"])
    result = {}
    for k in sample_grid:
        result[k] = df_in_window.groupby("serial_number").tail(k)
    return result


def build_gt_at_time(df: pd.DataFrame) -> pd.DataFrame:
    df_gt = df.loc[df.groupby('serial_number')['time'].transform('max') == df['time']]
    df_gt = df_gt.copy()
    df_gt["duration"] = df_gt["max_lifetime"] - df_gt['time']
    return df_gt[["serial_number", "duration", "failure"]]


def count_events_in_window(df: pd.DataFrame, t_start: float, t_end: float) -> int:
    events = df[
        (df["failure"] == 1) &
        (df["max_lifetime"] > t_start) &
        (df["max_lifetime"] <= t_end)
    ]
    return len(events["serial_number"].unique())


def evaluate_model_on_dataset(
    data_path: str,
    model_name: str,
    quantiles: List[int],
    sample_grid: List[int],
    metrics_list: List[str],
    quantile_mode: str,
    agg_dict,
    times: np.ndarray,
    models_folder: str,
    res_folder: str,
    data_ext: str,
    score_batchsize: int,
    max_obs_day: int,
    max_time: int,
) -> pd.DataFrame:
    if data_ext == "csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)

    scorer = ModelScorer()
    model_path = os.path.join(models_folder, model_name)
    if os.path.exists(model_path + "_model.pkl"):
        model = load_model(model_path + "_model.pkl")
    else:
        model = load_model(model_path + ".pkl")

    is_ddh = model_name.endswith('DDH')
    t_points = get_quantile_time_points(df, quantiles, mode=quantile_mode, max_obs_day=max_obs_day)
    results = []

    if quantile_mode == "equal_events":
        print("\n=== Event distribution across intervals ===")
        prev_t = 0
        for q, t_point in zip(quantiles, t_points):
            n_events = count_events_in_window(df, prev_t, t_point)
            print(f"Quantile {q:3d}: t={prev_t:7.2f} to {t_point:7.2f}, events={n_events:5d}")
            prev_t = t_point
        n_events = count_events_in_window(df, prev_t, max_obs_day)
        print(f"Quantile 100: t={prev_t:7.2f} to {max_obs_day:7.2f}, events={n_events:5d}")
        print("=" * 50 + "\n")

    predictions_dir = os.path.join(res_folder, "Predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    for q, t_point in zip(quantiles, t_points):
        print(f"Processing quantile {q} (t={t_point:.2f}, mode={quantile_mode})")
        histories_dict = build_history_at_time(df, t_point, sample_grid, max_time)
        X_gt = build_gt_at_time(histories_dict[sample_grid[0]])

        if is_ddh:
            X_hist = histories_dict[max(sample_grid)]
            if len(X_hist) == 0:
                continue
            dl = DataLoader(
                dataset=DataFrameDataset(X_hist, mode="score", times=times),
                batch_size=score_batchsize,
            )
            for k in sample_grid:
                print(f"  Processing agg_samples={k} for DDH")
                X_pred, _ = model.predict(dl, times=times, agg_horizon=k, trunc_right=max(sample_grid))
                X_pred.to_csv(f'{res_folder}/{model_name}_Pred_{q}.csv')
                X_gt.to_csv(f'{res_folder}/{model_name}_gt_{q}.csv')
                metrics = scorer.get_metrics(model, X_pred, X_gt, times, metrics=metrics_list)
                row = {
                    "quantile": q,
                    "t_point": t_point,
                    "agg_samples": k,
                    "agg_method": "DDH",
                    "agg_weight": -1,
                    "model_id": model_name,
                    "quantile_mode": quantile_mode,
                }
                for m in metrics:
                    row[m] = metrics[m]
                results.append(row)
        else:
            for k in sample_grid:
                X_hist = histories_dict[k]
                if len(X_hist) == 0:
                    continue
                dl = DataLoader(
                    dataset=DataFrameDataset(X_hist, mode="score", times=times),
                    batch_size=score_batchsize,
                )
                times_extended = get_extended_times(X_hist, agg_dict, times)
                X_pred, _ = model.predict(dl, times=times_extended)
                X_pred.to_csv(f'{predictions_dir}/{model_name}_Pred_{q}.csv')
                X_gt.to_csv(f'{predictions_dir}/{model_name}_gt_{q}.csv')
                for method in agg_dict:
                    for weight in agg_dict[method]:
                        aggregator = agg_dict[method][weight]
                        timeshift = X_pred.groupby('serial_number')['time'].transform('max') - X_pred['time']
                        aggregated_pred = aggregator.predict(X_pred, times, timeshift=timeshift)
                        metrics = scorer.get_metrics(
                            model, aggregated_pred, X_gt, times, metrics=metrics_list,
                        )
                        row = {
                            "quantile": q,
                            "t_point": t_point,
                            "agg_samples": k,
                            "agg_method": method,
                            "agg_weight": weight,
                            "model_id": model_name,
                            "quantile_mode": quantile_mode,
                        }
                        for m in metrics:
                            row[m] = metrics[m]
                        results.append(row)

    return pd.DataFrame(results)


def init_quantile_experiment_folder(cfg: QuantileAggConfig) -> str:
    res_folder = str(artifacts_dir(cfg.exp_num))
    models_folder = str(models_dir(cfg.exp_num))
    if os.path.exists(res_folder):
        raise ValueError("Experiment folder already exists")
    os.makedirs(models_folder)
    base_models = models_dir(cfg.base_exp_num)
    for m in os.listdir(base_models):
        if m.endswith(".pkl"):
            shutil.copy(base_models / m, models_folder)
    os.makedirs(os.path.join(res_folder, "Predictions"), exist_ok=True)
    return res_folder


def run_quantile_aggregation(cfg: QuantileAggConfig) -> None:
    np.random.seed(cfg.seed)
    times = np.arange(0, cfg.max_time)
    agg_dict = get_aggregators(cfg.aggregator_preset)
    res_folder = init_quantile_experiment_folder(cfg)
    models_folder = str(models_dir(cfg.exp_num))
    data_folder = resolve_path(cfg.data_folder)

    test_path = data_folder / f"{cfg.data_samples}_-1_test_preprocessed.{cfg.data_ext}"
    results_df = evaluate_model_on_dataset(
        data_path=str(test_path),
        model_name=cfg.model_name,
        quantiles=cfg.quantiles,
        sample_grid=cfg.sample_grid,
        metrics_list=cfg.metrics,
        quantile_mode=cfg.quantile_mode,
        agg_dict=agg_dict,
        times=times,
        models_folder=models_folder,
        res_folder=res_folder,
        data_ext=cfg.data_ext,
        score_batchsize=cfg.score_batchsize,
        max_obs_day=cfg.max_obs_day,
        max_time=cfg.max_time,
    )
    results_df.to_csv(os.path.join(res_folder, "results.csv"), index=False)
    print("Quantile aggregation experiment finished.")
