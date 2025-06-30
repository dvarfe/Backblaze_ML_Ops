# Программа, которая обучает модели и перебирает разные параметры
import copy
import pickle
import numpy as np
import time
from disk_analyzer.stages.model_scoring import ModelScorer
from torch.utils.data import DataLoader
from disk_analyzer.models.Dataset import DiskDataset
from Experiments import TIMES, TRAIN_BATCHSIZE
from PredictionsAggregator import PredictionsAggregator
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def write_dict(filename, dict_to_save):
    df = pd.DataFrame(dict_to_save)
    df.to_csv(filename, mode='a', header=False, index=False)


SCHEMA = {
    'train_samples': [],
    'test_samples': [],
    'agg_samples': [],
    'method': [],
    'hidden_dim': [],
    'agg_method': [],
    'agg_weight': [],
    'ci_train': [],
    'ibs_train': [],
    'ibs_bal_train': [],
    'model_id': []}


def create_res_file(filename):
    schema = SCHEMA
    df = pd.DataFrame(schema)
    df.to_csv(filename, index=False)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def sample_first_observations(df, sample_grid):
    dict_of_all_samples = {}
    df = df.sort_values(by=['serial_number', 'time'])
    for first_n in sample_grid:
        df_train_sampled = df.groupby('serial_number').head(max(sample_grid)).groupby('serial_number').tail(first_n)
        df_train_sampled = df_train_sampled.drop_duplicates().sort_values(
            by=['serial_number', 'time'])  # .reset_index(drop=True)
        dict_of_all_samples[first_n] = df_train_sampled
    return dict_of_all_samples


DATASET_TRAIN_SAMPLES = 30
MODEL_TRAIN_SAMPLES = 10
SAMPLE_GRID = np.arange(1, 10)
RESPATH = f'Artifacts/Exp_22/Agg_{DATASET_TRAIN_SAMPLES}_{MODEL_TRAIN_SAMPLES}_{max(SAMPLE_GRID)}.csv'
TEST_GRID = [10]
MODELS_SIZE = [128, 512]

if __name__ == "__main__":
    """
    Я переделал Predictions Aggregator так, чтобы теперь он только агрегировал уже готовые прогнозы.
    Для этого нужно получить extended шкалу, а потом по ней построить прогноз моделью.
    Прогноз мы строим один раз, а затем нарезаем его на кусочки разной длины(от 8 до 10, от 7 до 10... от 1 до 10 наблюдения)
    с помощью sample_first_observation.
    Эти кусочки уже агрегируем, передавая их в pred_agg.
    """
    grid_search = pd.read_csv('Artifacts/Exp_22\grid_search.csv')
    grid_search = grid_search[(grid_search['train_samples'] == MODEL_TRAIN_SAMPLES) & (grid_search['error'] != 1)]
    create_res_file(RESPATH)
    models = grid_search[grid_search['hidden_dim'].isin(MODELS_SIZE)]['model_id'].unique()
    for test_samples in TEST_GRID:
        df_test = pd.read_csv(f'Preprocessed/{DATASET_TRAIN_SAMPLES}_{test_samples}_test_preprocessed.csv')
        # Выкидываем последние наблюдения в каждой серии
        df_test = df_test[df_test['time'] != df_test['max_lifetime']]
        df_test = df_test.sort_values(by=['serial_number', 'time'])
        dl_test = DataLoader(
            dataset=DiskDataset(
                'score', [f'Preprocessed/{DATASET_TRAIN_SAMPLES}_{test_samples}_test_preprocessed.csv']),
            batch_size=TRAIN_BATCHSIZE)

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
        aggregators_dict = {
            "n_dist": {
                "0.01": PredictionsAggregator(mode='n_dist', weight=0.01),
                "0.1": PredictionsAggregator(mode='n_dist', weight=0.1),
                "0.3": PredictionsAggregator(mode='n_dist', weight=0.3),
                "0.5": PredictionsAggregator(mode='n_dist', weight=0.5),
                "0.7": PredictionsAggregator(mode='n_dist', weight=0.7),
                "0.9": PredictionsAggregator(mode='n_dist', weight=0.9),
                "0.99": PredictionsAggregator(mode='n_dist', weight=0.99)
            }
        }
        any_dict_key = list(aggregators_dict.keys())[0]
        any_dict_key_key = list(aggregators_dict[any_dict_key].keys())[0]
        times_extended = aggregators_dict[any_dict_key][any_dict_key_key].get_extended_times(df_test, times=TIMES)
        for model_id in models:
            model = load_model(f"Artifacts/Exp_22/models/{model_id}_model.pkl")
            X_pred, X_gt = model.predict(dl_test, times=times_extended)
            X_pred = X_pred.sort_values(by=['serial_number', 'time'])
            X_gt = X_gt.sort_values(by=['serial_number', 'time'])

            sampled_predictions = sample_first_observations(X_pred, SAMPLE_GRID)

            scorer = ModelScorer()

            for n_samples in SAMPLE_GRID:
                if n_samples > test_samples:
                    continue
                h_dim = grid_search[grid_search['model_id'] == model_id]['hidden_dim'].unique()[0]
                print(f'Начало обработки {n_samples} n_samples, {h_dim} hidden_dim')
                time_start = time.time()
                statistics = copy.deepcopy(SCHEMA)

                statistics['train_samples'] = MODEL_TRAIN_SAMPLES
                statistics['test_samples'] = test_samples
                statistics['agg_samples'] = [n_samples]
                statistics['method'] = [None]
                statistics['hidden_dim'] = h_dim
                statistics['ci_train'] = [None]
                statistics['ibs_train'] = [None]
                statistics['ibs_bal_train'] = [None]
                statistics['model_id'] = [model_id]

                cur_X_pred = sampled_predictions[n_samples]
                cur_timeshift = cur_X_pred.groupby('serial_number')['time'].transform('max') - cur_X_pred['time']
                # Оставляем для метрики последнее наблюдение к которому агрегируемся. Вообще-то это можно сделать один раз.
                # Просто занести в sample... но потом всё потом. TODO
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

                        cur_statistics['ci_train'], cur_statistics['ibs_train'], cur_statistics['ibs_bal_train'] = scorer.get_ci_ibs_ibs_bal(
                            model, df_pred=aggregated_pred, df_gt=cur_X_gt, times=TIMES)

                        write_dict(RESPATH, cur_statistics)
                print(f'Обработка {n_samples} завершена за {time.time() - time_start} секунд')
