# Программа, которая обучает модели и перебирает разные параметры
import copy
import pickle
import time

import os  # noqa
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # noqa

import shutil
from typing import List
import pandas as pd
import numpy as np

from disk_analyzer.stages.model_scoring import ModelScorer
from torch.utils.data import DataLoader
from disk_analyzer.models.Dataset import DiskDataset
from Experiments import TIMES, TRAIN_BATCHSIZE
from PredictionsAggregator import HazardSumAgg

np.random.seed(42)


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
            by=['serial_number', 'time'])
        dict_of_all_samples[first_n] = df_train_sampled
    return dict_of_all_samples


DATASET_TRAIN_SAMPLES = 30
MODEL_TRAIN_SAMPLES = 20
SAMPLE_GRID = np.arange(3, 4)
EXP_NUM = 54
BASE_EXP_NUM = 46  # Exp number from which fitted models are taken
RES_FOLDER = os.path.join("Artifacts", f"Exp_{EXP_NUM}")
MODELS_FOLDER = os.path.join(RES_FOLDER, "models")
RES_FILENAME = os.path.join(
    RES_FOLDER, f"Agg_{DATASET_TRAIN_SAMPLES}_{MODEL_TRAIN_SAMPLES}_{max(SAMPLE_GRID)}.csv")
TEST_GRID = [10]
MODELS_SIZE = [2048]


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


if __name__ == "__main__":

    grid_search = pd.read_csv(f'{RES_FOLDER}/grid_search.csv')
    grid_search = grid_search[(grid_search['train_samples'] == MODEL_TRAIN_SAMPLES) & (grid_search['error'] != 1)]

    models = grid_search[grid_search['hidden_dim'].isin(MODELS_SIZE)]['model_id'].unique()
    for test_samples in TEST_GRID:
        test_path = f'Preprocessed_new/{DATASET_TRAIN_SAMPLES}_{test_samples}_test_preprocessed.csv'

        df_test = pd.read_csv(test_path)
        # Выкидываем последние наблюдения в каждой серии
        df_test = df_test[df_test['time'] != df_test['max_lifetime']]
        df_test = df_test.sort_values(by=['serial_number', 'time'])
        dl_test = DataLoader(
            dataset=DiskDataset(
                'score', [test_path]),
            batch_size=TRAIN_BATCHSIZE)

        for model_id in models:
            model = load_model(f"{MODELS_FOLDER}/{model_id}_model.pkl")
            pred_agg = HazardSumAgg()
            times_extended = TIMES
            X_pred, X_gt = model.predict(dl_test, times=times_extended)
            X_pred = X_pred.sort_values(by=['serial_number', 'time'])
            X_gt = X_gt.sort_values(by=['serial_number', 'time'])

            sampled_predictions = sample_first_observations(X_pred, SAMPLE_GRID)

            scorer = ModelScorer()

            for n_samples in SAMPLE_GRID:
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
                cur_timeshift = pred_agg.get_timeshift(cur_X_pred)

                cur_X_gt = X_gt.loc[cur_X_pred.index, :]
                cur_X_gt = cur_X_gt[cur_X_gt['time'] == cur_X_gt.groupby('serial_number')['time'].transform('max')]
                aggregated_pred = pred_agg.predict(cur_X_pred, TIMES, timeshift=cur_timeshift)
                print(f'Обработка {n_samples} завершена за {time.time() - time_start} секунд')
