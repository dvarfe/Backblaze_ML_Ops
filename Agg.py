# Программа, которая обучает модели и перебирает разные параметры
import copy
import pickle
import time

import os  # noqa
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # noqa

import shutil
from typing import List, Dict
import pandas as pd
import numpy as np

from disk_analyzer.stages.model_scoring import ModelScorer
from torch.utils.data import DataLoader
from disk_analyzer.models.Dataset import DiskDataset
from Experiments import TIMES, TRAIN_BATCHSIZE
from PredictionsAggregator import PredictionsAggregator

np.random.seed(42)


def create_schema(base_schema, metrics_list, model_name):
    ret_schema = copy.deepcopy(base_schema)
    for metric in metrics_list:
        ret_schema[f'{metric}_test'] = []
    if model_name == 'Cox':
        ret_schema['l1_ratio'] = []
        ret_schema['penalizer'] = []
    elif model_name == 'SP':
        ret_schema['h_dim'] = []
    return ret_schema


BASE_SCHEMA: Dict[str, list] = {
    'train_samples': [],
    'test_samples': [],
    'agg_samples': [],
    'method': [],
    'agg_method': [],
    'agg_weight': [],
    'model_id': []
}

METRICS_LIST = {'ci', 'ibs', 'iauc'}
DATASET_TRAIN_SAMPLES = 30
MODEL_TRAIN_SAMPLES = 20
SAMPLE_GRID = np.arange(1, 10)
EXP_NUM = 69
BASE_EXP_NUM = 68  # Exp number from which fitted models are taken
DATA_FOLDER = "Preprocessed_new"
RES_FOLDER = os.path.join("Artifacts", f"Exp_{EXP_NUM}")
MODELS_FOLDER = os.path.join(RES_FOLDER, "models")
RES_FILENAME = os.path.join(
    RES_FOLDER, f"Agg_{DATASET_TRAIN_SAMPLES}_{MODEL_TRAIN_SAMPLES}_{max(SAMPLE_GRID)}.csv")
TEST_GRID = [10]
# MODELS_SIZE = [2048]
MODEL_NAME = 'Cox'
SCHEMA = create_schema(BASE_SCHEMA, METRICS_LIST, MODEL_NAME)


def create_res_file(filename):
    schema = SCHEMA
    df = pd.DataFrame(schema)
    df.to_csv(filename, index=False)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_dict(filename, dict_to_save):
    df = pd.DataFrame(dict_to_save)
    df.to_csv(filename, mode='a', header=False, index=False)


def sample_first_observations(df, sample_grid):
    dict_of_all_samples = {}
    df = df.sort_values(by=['serial_number', 'time'])
    for first_n in sample_grid:
        df_train_sampled = df.groupby('serial_number').head(max(sample_grid)).groupby('serial_number').tail(first_n)
        df_train_sampled = df_train_sampled.drop_duplicates().sort_values(
            by=['serial_number', 'time'])
        dict_of_all_samples[first_n] = df_train_sampled
    return dict_of_all_samples


def get_models_from_exp(base_exp_num: int) -> List[str]:
    base_exp_folder = os.path.join("Artifacts", f'Exp_{base_exp_num}', f'models')
    return [os.path.join(base_exp_folder, m) for m in os.listdir(base_exp_folder) if m.endswith(".pkl")]


def copy_models(models_path: List[str], dest_folder: str):
    for model in models_path:
        if not os.path.exists(model):
            raise ValueError("Model doesn't exist")
        shutil.copy(model, dest_folder)


def init_experiments_folder(base_exp_num):
    if os.path.exists(RES_FILENAME):
        raise ValueError('Path exists!')
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
    models_path = get_models_from_exp(base_exp_num)
    copy_models(models_path, MODELS_FOLDER)
    create_res_file(RES_FILENAME)
    shutil.copy(f"Artifacts/Exp_{BASE_EXP_NUM}/grid_search.csv", RES_FOLDER)


# def get_X_gt_with_na_short_narezka(df):
#     df = df.sort_values(['serial_number', 'time'])
#     X_gt = df.copy()
#     X_gt['next_time'] = X_gt.groupby('id')['time'].shift(-1)
#     X_gt['event'] = X_gt.groupby('id')['event'].shift(-1)
#     X_gt['duration'] = X_gt['next_time'] - X_gt['time']
#     X_gt = X_gt[['time', 'id', 'duration', 'event']]

#     return X_gt Неправильно, на новых данных надо проставить цензуру.


def get_X_gt_long_narezka(df):
    """Функция для подготовки ground truth данных"""
    df = df.sort_values(['serial_number', 'time'])
    X_gt = df.copy()
    X_gt['event_time'] = X_gt['max_lifetime']
    X_gt['duration'] = X_gt['event_time'] - X_gt['time']
    X_gt = X_gt.loc[X_gt['duration'] != 0, ['time', 'serial_number', 'duration', 'failure']]
    return X_gt


if __name__ == "__main__":
    """
    Я переделал Predictions Aggregator так, чтобы теперь он только агрегировал уже готовые прогнозы.
    Для этого нужно получить extended шкалу, а потом по ней построить прогноз моделью.
    Прогноз мы строим один раз, а затем нарезаем его на кусочки разной длины(от 8 до 10, от 7 до 10... от 1 до 10 наблюдения)
    с помощью sample_first_observation.
    Эти кусочки уже агрегируем, передавая их в pred_agg.
    """

    init_experiments_folder(BASE_EXP_NUM)

    grid_search = pd.read_csv(f'{RES_FOLDER}/grid_search.csv')
    grid_search = grid_search[(grid_search['train_samples'] == MODEL_TRAIN_SAMPLES) &
                              (grid_search['error'] != 1) & (grid_search['test_samples'] == 1)]

    for test_samples in TEST_GRID:
        test_path = os.path.join(DATA_FOLDER, f'{DATASET_TRAIN_SAMPLES}_{test_samples}_test_preprocessed.csv')
        df_test = pd.read_csv(test_path)
        # Выкидываем последние наблюдения в каждой серии
        df_test = df_test[df_test['time'] != df_test['max_lifetime']]
        df_test = df_test.sort_values(by=['serial_number', 'time'])
        dl_test = DataLoader(
            dataset=DiskDataset(
                'score', [test_path]),
            batch_size=TRAIN_BATCHSIZE)

        aggregators_dict = {
            "n_dist": {
                "0.01": PredictionsAggregator(mode='n_dist', weight=0.01),
                "0.1": PredictionsAggregator(mode='n_dist', weight=0.1),
                "0.3": PredictionsAggregator(mode='n_dist', weight=0.3),
                "0.5": PredictionsAggregator(mode='n_dist', weight=0.5),
                "0.7": PredictionsAggregator(mode='n_dist', weight=0.7),
                "0.9": PredictionsAggregator(mode='n_dist', weight=0.9),
                "0.99": PredictionsAggregator(mode='n_dist', weight=0.99)
            },
            "t_dist": {
                "0.1": PredictionsAggregator(mode='t_dist', weight=0.1),
                "1": PredictionsAggregator(mode='t_dist', weight=1),
                "10": PredictionsAggregator(mode='t_dist', weight=10),
                "25": PredictionsAggregator(mode='t_dist', weight=25),
                "50": PredictionsAggregator(mode='t_dist', weight=50),
                "100": PredictionsAggregator(mode='t_dist', weight=100),
                "1000": PredictionsAggregator(mode='t_dist', weight=1000)
            },
            "prob_dist": {
                "-1": PredictionsAggregator(mode='prob_dist'),
            },
            "geom": {
                "0.01": PredictionsAggregator(mode='geom', weight=0.01),
                "0.1": PredictionsAggregator(mode='geom', weight=0.1),
                "0.3": PredictionsAggregator(mode='geom', weight=0.3),
                "0.5": PredictionsAggregator(mode='geom', weight=0.5),
                "0.7": PredictionsAggregator(mode='geom', weight=0.7),
                "0.9": PredictionsAggregator(mode='geom', weight=0.9),
                "0.99": PredictionsAggregator(mode='geom', weight=0.99)
            },
        }

        any_dict_key = list(aggregators_dict.keys())[0]
        any_dict_key_key = list(aggregators_dict[any_dict_key].keys())[0]
        times_extended = aggregators_dict[any_dict_key][any_dict_key_key].get_extended_times(df_test, times=TIMES)
        for _, row in grid_search.iterrows():
            model_id = row['model_id']
            hparams = {}
            if MODEL_NAME == 'Cox':
                hparams = {'l1_ratio': row['l1_ratio'],
                           'penalizer': row['penalizer']}
            elif MODEL_NAME == 'SP':
                hparams = {'h_dim': row['h_dim']}

            model = load_model(os.path.join(MODELS_FOLDER, f'{model_id}_model.pkl'))
            X_pred, X_gt = model.predict(dl_test, times=times_extended)
            X_pred = X_pred.sort_values(by=['serial_number', 'time'])
            X_gt = X_gt.sort_values(by=['serial_number', 'time'])

            if 'iauc' in METRICS_LIST:
                train_samples = row['train_samples']
                df_train = pd.read_csv(os.path.join(DATA_FOLDER, f'{train_samples}_train_preprocessed.csv'))
                df_train_gt = get_X_gt_long_narezka(df_train)
            else:
                df_train_gt = None

            sampled_predictions = sample_first_observations(X_pred, SAMPLE_GRID)

            scorer = ModelScorer()

            for n_samples in SAMPLE_GRID:
                if n_samples > test_samples:
                    continue
                print(f'Начало обработки {n_samples} n_samples, {hparams}')
                time_start = time.time()
                statistics = copy.deepcopy(SCHEMA)

                statistics['train_samples'] = MODEL_TRAIN_SAMPLES
                statistics['test_samples'] = test_samples
                statistics['agg_samples'] = [n_samples]
                statistics['method'] = [None]
                statistics['model_id'] = [model_id]
                for key in hparams:
                    statistics[key] = hparams[key]

                cur_X_pred = sampled_predictions[n_samples]
                cur_timeshift = cur_X_pred.groupby('serial_number')['time'].transform('max') - cur_X_pred['time']
                # Оставляем для метрики последнее наблюдение к которому агрегируемся. Вообще-то это можно сделать один раз.
                # Просто занести в sample... но потом, всё потом. TODO
                cur_X_gt = X_gt.loc[cur_X_pred.index, :]
                cur_X_gt = cur_X_gt[cur_X_gt['time'] == cur_X_gt.groupby('serial_number')['time'].transform('max')]

                for method in aggregators_dict:
                    for weight in aggregators_dict[method]:
                        cur_statistics = copy.deepcopy(statistics)
                        cur_statistics['method'] = model_id.split('_')[1]
                        cur_statistics['agg_method'] = method
                        cur_statistics['agg_weight'] = weight

                        cur_pred_agg = aggregators_dict[method][weight]
                        aggregated_pred = cur_pred_agg.predict(cur_X_pred, TIMES, timeshift=cur_timeshift)

                        metrics = scorer.get_metrics(
                            model, aggregated_pred, cur_X_gt, TIMES,
                            metrics=METRICS_LIST,
                            df_train=df_train_gt)
                        for metric in metrics:
                            cur_statistics[f'{metric}_test'] = metrics.get(metric)

                        write_dict(RES_FILENAME, cur_statistics)
                print(f'Обработка {n_samples} завершена за {time.time() - time_start} секунд')

        # aggregators_dict = {
        #     "t_dist": {
        #         "0.1": PredictionsAggregator(mode='t_dist', weight=0.1),
        #         "1": PredictionsAggregator(mode='t_dist', weight=1),
        #         "10": PredictionsAggregator(mode='t_dist', weight=10),
        #         "25": PredictionsAggregator(mode='t_dist', weight=25),
        #         "50": PredictionsAggregator(mode='t_dist', weight=50),
        #         "100": PredictionsAggregator(mode='t_dist', weight=100),
        #         "1000": PredictionsAggregator(mode='t_dist', weight=1000)
        #     }
        # }

        # aggregators_dict = {
        #     "geom": {
        #         "0.01": PredictionsAggregator(mode='geom', weight=0.01),
        #         "0.1": PredictionsAggregator(mode='geom', weight=0.1),
        #         "0.3": PredictionsAggregator(mode='geom', weight=0.3),
        #         "0.5": PredictionsAggregator(mode='geom', weight=0.5),
        #         "0.7": PredictionsAggregator(mode='geom', weight=0.7),
        #         "0.9": PredictionsAggregator(mode='geom', weight=0.9),
        #         "0.99": PredictionsAggregator(mode='geom', weight=0.99)
        #     }
        # }
        # aggregators_dict = {
        #     "n_dist": {
        #         "0.01": PredictionsAggregator(mode='n_dist', weight=0.01),
        #         "0.1": PredictionsAggregator(mode='n_dist', weight=0.1),
        #         "0.3": PredictionsAggregator(mode='n_dist', weight=0.3),
        #         "0.5": PredictionsAggregator(mode='n_dist', weight=0.5),
        #         "0.7": PredictionsAggregator(mode='n_dist', weight=0.7),
        #         "0.9": PredictionsAggregator(mode='n_dist', weight=0.9),
        #         "0.99": PredictionsAggregator(mode='n_dist', weight=0.99)
        #     }
        # }
