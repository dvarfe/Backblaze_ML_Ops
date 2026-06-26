"""Aggregation grid search over trained models."""

import copy
import os
import shutil
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.research.aggregation.presets import get_aggregators
from disk_analyzer.research.config import DEFAULT_TIMES, AggConfig
from disk_analyzer.research.io import load_model, save_aggregation_config
from disk_analyzer.research.paths import aggregation_dir, artifacts_dir, models_dir, resolve_path
from disk_analyzer.stages.model_scoring import ModelScorer


BASE_SCHEMA: Dict[str, list] = {
    'train_samples': [],
    'agg_samples': [],
    'method': [],
    'agg_method': [],
    'agg_weight': [],
    'model_id': [],
}


def create_agg_schema(metrics_list: List[str]) -> Dict[str, list]:
    schema = copy.deepcopy(BASE_SCHEMA)
    for metric in metrics_list:
        schema[f'{metric}_test'] = []
    return schema


def sample_first_observations(df, sample_grid):
    dict_of_all_samples = {}
    df = df.sort_values(by=['serial_number', 'time'])
    for first_n in sample_grid:
        df_train_sampled = df.groupby('serial_number').head(max(sample_grid)).groupby('serial_number').tail(first_n)
        df_train_sampled = df_train_sampled.drop_duplicates().sort_values(by=['serial_number', 'time'])
        dict_of_all_samples[first_n] = df_train_sampled
    return dict_of_all_samples


def get_extended_times(df, agg_dict, times):
    any_dict_key = list(agg_dict.keys())[0]
    any_dict_key_key = list(agg_dict[any_dict_key].keys())[0]
    return agg_dict[any_dict_key][any_dict_key_key].get_extended_times(df, times)


def calculate_metrics(
    model, predictions_aggregator, scorer, method, weight, n_samples,
    timeshift, X, X_gt, times, model_train_samples, model_name,
    metric_postfix='test', metrics_list=None, df_train_gt=None,
):
    if metrics_list is None:
        metrics_list = ['ci', 'ibs']
    aggregated_pred = predictions_aggregator.predict(X, times, timeshift=timeshift)
    metrics = scorer.get_metrics(
        model, aggregated_pred, X_gt, times,
        metrics=metrics_list,
        df_train=df_train_gt,
    )
    result_row = {
        'train_samples': model_train_samples,
        'agg_samples': n_samples,
        'method': model_name.split('_')[1] if '_' in model_name else model_name,
        'agg_method': method,
        'agg_weight': weight,
        'model_id': model_name,
    }
    for metric in metrics:
        result_row[f'{metric}_{metric_postfix}'] = metrics.get(metric)
    return result_row


def eval_model(
    data_path,
    model_name,
    agg_dict,
    metrics_list,
    times,
    train_batchsize,
    data_ext,
    sample_grid,
    models_folder,
    model_train_samples,
    metric_postfix='test',
    data_folder=None,
):
    if data_ext == 'csv':
        df_data = pd.read_csv(data_path)
    else:
        df_data = pd.read_parquet(data_path)

    df_data = df_data[df_data['time'] != df_data['max_lifetime']]
    df_data = df_data.sort_values(by=['serial_number', 'time'])

    dl_data = DataLoader(
        dataset=DiskDataset('score', [data_path]),
        batch_size=train_batchsize,
    )
    times_extended = get_extended_times(df_data, agg_dict, times)

    if 'iauc' in metrics_list and data_folder is not None:
        train_path = os.path.join(data_folder, f'1_train_preprocessed.{data_ext}')
        if data_ext == 'csv':
            df_train = pd.read_csv(train_path)
        else:
            df_train = pd.read_parquet(train_path)
        df_train_gt = df_train.copy()
        df_train_gt['duration'] = df_train_gt['max_lifetime'] - df_train_gt['time']
        df_train_gt = df_train_gt.loc[df_train_gt['duration'] != 0, ['time', 'serial_number', 'duration', 'failure']]
    else:
        df_train_gt = None

    model_path = os.path.join(models_folder, model_name)
    if os.path.exists(f'{model_path}_model.pkl'):
        model = load_model(f'{model_path}_model.pkl')
    else:
        model = load_model(f'{model_path}.pkl')

    if model_name.endswith('DDH'):
        scorer = ModelScorer()
        results_list = []
        for n_samples in sample_grid:
            print(f'Начало обработки {n_samples} n_samples')
            time_start = time.time()
            X_pred, X_gt = model.predict(dl_data, times=times, agg_horizon=n_samples, trunc_right=max(sample_grid))
            metrics = scorer.get_metrics(
                model, X_pred, X_gt, times,
                metrics=metrics_list,
                df_train=df_train_gt,
            )
            result_row = {
                'train_samples': model_train_samples,
                'agg_samples': n_samples,
                'method': model_name.split('_')[1] if '_' in model_name else 'DDH',
                'agg_method': 'DDH',
                'agg_weight': -1,
                'model_id': model_name,
            }
            for metric in metrics:
                result_row[f'{metric}_{metric_postfix}'] = metrics.get(metric)
            results_list.append(result_row)
            print(f'Обработка {n_samples} завершена за {time.time() - time_start} секунд')
    else:
        X_pred, X_gt = model.predict(dl_data, times=times_extended)
        X_pred = X_pred.sort_values(by=['serial_number', 'time'])
        X_gt = X_gt.sort_values(by=['serial_number', 'time'])
        sampled_predictions = sample_first_observations(X_pred, sample_grid)
        scorer = ModelScorer()
        results_list = []
        for n_samples in sample_grid:
            print(f'Начало обработки {n_samples} n_samples')
            time_start = time.time()
            cur_X_pred = sampled_predictions[n_samples]
            cur_timeshift = cur_X_pred.groupby('serial_number')['time'].transform('max') - cur_X_pred['time']
            cur_X_gt = X_gt.loc[cur_X_pred.index, :]
            cur_X_gt = cur_X_gt[cur_X_gt['time'] == cur_X_gt.groupby('serial_number')['time'].transform('max')]
            for method in agg_dict:
                for weight in agg_dict[method]:
                    result_row = calculate_metrics(
                        model, agg_dict[method][weight], scorer, method, weight, n_samples,
                        cur_timeshift, cur_X_pred, cur_X_gt, times,
                        model_train_samples, model_name,
                        metric_postfix, metrics_list, df_train_gt,
                    )
                    results_list.append(result_row)
            print(f'Обработка {n_samples} завершена за {time.time() - time_start} секунд')

    return pd.DataFrame(results_list)


def prepare_agg_folder(
    agg_folder: str,
    res_filename: str,
    schema: Dict[str, list],
    exp_desc: str,
    base_exp_num: int,
) -> None:
    os.makedirs(agg_folder, exist_ok=True)
    if not os.path.exists(res_filename):
        pd.DataFrame(schema).to_csv(res_filename, index=False)
        desc_path = os.path.join(agg_folder, "Description.txt")
        with open(desc_path, "w") as f:
            f.write(exp_desc)
    grid_search_dst = os.path.join(agg_folder, "grid_search.csv")
    if not os.path.exists(grid_search_dst):
        grid_search_src = artifacts_dir(base_exp_num) / "grid_search.csv"
        if grid_search_src.exists():
            try:
                shutil.copy(grid_search_src, agg_folder)
            except OSError:
                pass


def append_agg_results(res_filename: str, new_df: pd.DataFrame, model_id: str) -> None:
    if os.path.exists(res_filename):
        existing = pd.read_csv(res_filename)
        if not existing.empty and "model_id" in existing.columns:
            existing = existing[existing["model_id"] != model_id]
        results_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        results_df = new_df
    results_df.to_csv(res_filename, index=False)


def run_aggregation(cfg: AggConfig) -> None:
    np.random.seed(42)
    times = cfg.times if cfg.times is not None else DEFAULT_TIMES
    agg_dict = get_aggregators(cfg.aggregator_preset)
    schema = create_agg_schema(cfg.metrics)

    data_folder = str(resolve_path(cfg.data_folder))
    models_folder = str(models_dir(cfg.base_exp_num))
    agg_folder = str(aggregation_dir(cfg.base_exp_num))
    res_filename = os.path.join(
        agg_folder,
        f"Agg_{cfg.dataset_train_samples}_{cfg.model_train_samples}_{max(cfg.sample_grid)}.csv",
    )

    prepare_agg_folder(agg_folder, res_filename, schema, cfg.exp_desc, cfg.base_exp_num)

    if cfg.config_source_path:
        save_aggregation_config(cfg.base_exp_num, cfg.config_source_path, cfg.model_name)

    train_path = os.path.join(data_folder, f'{cfg.dataset_train_samples}_train_preprocessed.{cfg.data_ext}')
    train_res = eval_model(
        data_path=train_path,
        model_name=cfg.model_name,
        agg_dict=agg_dict,
        metrics_list=cfg.metrics,
        times=times,
        train_batchsize=cfg.train_batchsize,
        data_ext=cfg.data_ext,
        sample_grid=cfg.sample_grid,
        models_folder=models_folder,
        model_train_samples=cfg.model_train_samples,
        metric_postfix='train',
        data_folder=data_folder,
    )

    test_sample = cfg.test_samples[0]
    test_path = os.path.join(
        data_folder,
        f'{cfg.dataset_train_samples}_{test_sample}_test_preprocessed.{cfg.data_ext}',
    )
    test_res = eval_model(
        data_path=test_path,
        model_name=cfg.model_name,
        agg_dict=agg_dict,
        metrics_list=cfg.metrics,
        times=times,
        train_batchsize=cfg.train_batchsize,
        data_ext=cfg.data_ext,
        sample_grid=cfg.sample_grid,
        models_folder=models_folder,
        model_train_samples=cfg.model_train_samples,
        metric_postfix='test',
        data_folder=data_folder,
    )

    results_df = pd.merge(
        train_res, test_res, how='outer',
        on=['agg_samples', 'train_samples', 'method', 'agg_method', 'agg_weight', 'model_id'],
    )
    append_agg_results(res_filename, results_df, cfg.model_name)
    print(f"Aggregation results saved to {res_filename}")
