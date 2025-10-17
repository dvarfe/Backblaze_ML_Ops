import copy
import pickle
import time
import os

import shutil
from typing import List, Dict
import pandas as pd
import numpy as np

from disk_analyzer.stages.model_scoring import ModelScorer
from torch.utils.data import DataLoader
from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.models.MagiCox import MagiCoxTimeVaryingEstimator
from Experiments import TIMES, TRAIN_BATCHSIZE

np.random.seed(42)

# === КОНСТАНТЫ ===
BASE_EXP = 68  # Эксперимент, из которого берем обученные модели и grid_search.csv
EXP_NUM = 72   # Новый номер эксперимента для сохранения результатов оценки
DATA_FOLDER = "Preprocessed_new"
RES_FOLDER = os.path.join("Artifacts", f"Exp_{EXP_NUM}")
MODELS_FOLDER = os.path.join("Artifacts", f"Exp_{BASE_EXP}", "models")
RES_FILENAME = os.path.join(
    RES_FOLDER, f"eval_grid_search.csv"
)
TEST_GRID = [1, 10]
METRICS_LIST = {'ci', 'ibs'}
MODEL_NAME = 'Cox'

BASE_SCHEMA: Dict[str, list] = {
    'train_samples': [],
    'test_samples': [],
    'method': [],
    'error': [],
    'error_text': [],
    'model_id': []
}


def create_schema(base_schema, metrics_list, model_name):
    ret_schema = copy.deepcopy(base_schema)
    for metric in metrics_list:
        ret_schema[f'{metric}_train_same_size'] = []
        ret_schema[f'{metric}_train_max_size'] = []
        ret_schema[f'{metric}_test'] = []
    if model_name == 'Cox':
        ret_schema['l1_ratio'] = []
        ret_schema['penalizer'] = []
    elif model_name == 'SP':
        ret_schema['h_dim'] = []
    return ret_schema


SCHEMA = create_schema(BASE_SCHEMA, METRICS_LIST, MODEL_NAME)
TRAIN_GRID = [1, 2, 5, 10, 15, 20, 30, 40, 50]


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


if __name__ == "__main__":
    """
    Скрипт, который берет обученные модели из другого эксперимента (BASE_EXP)
    и для каждой модели пересчитывает метрики на трейне и тесте
    по тем же правилам, по которым считались в оригинальном эксперименте.
    """
    if not os.path.exists(RES_FOLDER):
        os.makedirs(RES_FOLDER)
    if os.path.exists(RES_FILENAME):
        raise ValueError('Path exists!')
    create_res_file(RES_FILENAME)

    # Загружаем grid_search.csv из BASE_EXP
    grid_search = pd.read_csv(os.path.join("Artifacts", f"Exp_{BASE_EXP}", "grid_search.csv"))
    grid_search = grid_search[(grid_search['l1_ratio'] == grid_search['l1_ratio'].unique()[2]) & (
        grid_search['penalizer'] == grid_search['penalizer'].unique()[2])]  # !!!!!
    scorer = ModelScorer()

    # Для train_max_metrics нужен этот датафрейм
    df_train_max = pd.read_csv(f'{DATA_FOLDER}/{max(TRAIN_GRID)}_train_preprocessed.csv')
    df_train_max['duration'] = df_train_max['max_lifetime'] - df_train_max['time']
    df_train_max = df_train_max[['duration', 'failure', 'time']]
    dl_score_max = DataLoader(
        dataset=DiskDataset('score', [f'{DATA_FOLDER}/{max(TRAIN_GRID)}_train_preprocessed.csv']),
        batch_size=TRAIN_BATCHSIZE,
    )

    for idx, row in grid_search.iterrows():
        try:
            train_samples = int(row['train_samples'])
            method = row['method']
            model_id = row['model_id']
            l1_ratio = row.get('l1_ratio', None)
            penalizer = row.get('penalizer', None)
            # hidden_dim = row.get('hidden_dim', None)

            hparams = {}
            if MODEL_NAME == 'Cox':
                hparams = {'l1_ratio': l1_ratio, 'penalizer': penalizer}
            elif MODEL_NAME == 'SP':
                hparams = {'h_dim': hidden_dim}

            model = load_model(os.path.join(MODELS_FOLDER, f'{model_id}_model.pkl'))
            model.__class__ = MagiCoxTimeVaryingEstimator  # !!!!!!!!!!!!!!!!!!!

            statistics = copy.deepcopy(SCHEMA)
            statistics['train_samples'] = [train_samples]
            statistics['method'] = [method]
            # statistics['hidden_dim'] = [hidden_dim]
            statistics['l1_ratio'] = [l1_ratio]
            statistics['penalizer'] = [penalizer]
            statistics['error'] = [0]
            statistics['error_text'] = ['']
            statistics['model_id'] = [model_id]

            # TRAIN SAME SIZE
            dl_train_score = DataLoader(
                dataset=DiskDataset('score', [f'{DATA_FOLDER}/{train_samples}_train_preprocessed.csv']),
                batch_size=TRAIN_BATCHSIZE)
            df_train_predictions, df_train_gt = model.predict(dl_train_score, TIMES)
            train_metrics = scorer.get_metrics(
                model, df_train_predictions, df_train_gt, TIMES,
                metrics=METRICS_LIST,
                df_train=df_train_max
            )
            for metric in METRICS_LIST:
                statistics[f'{metric}_train_same_size'] = [train_metrics.get(metric)]

            # TRAIN MAX SIZE
            df_train_max_predictions, df_train_max_gt = model.predict(dl_score_max, TIMES)
            train_max_metrics = scorer.get_metrics(
                model, df_train_max_predictions, df_train_max_gt, TIMES,
                metrics=METRICS_LIST,
                df_train=df_train_max
            )
            for metric in METRICS_LIST:
                statistics[f'{metric}_train_max_size'] = [train_max_metrics.get(metric)]

            for test_samples in TEST_GRID:
                cur_statistics = copy.deepcopy(statistics)
                cur_statistics['test_samples'] = [test_samples]
                try:
                    dl_test_score = DataLoader(
                        dataset=DiskDataset(
                            'score', [f'{DATA_FOLDER}/{train_samples}_{test_samples}_test_preprocessed.csv']),
                        batch_size=TRAIN_BATCHSIZE)
                    df_test_pred, df_test_pred_gt = model.predict(dl_test_score, TIMES)
                    test_metrics = scorer.get_metrics(
                        model, df_test_pred, df_test_pred_gt, TIMES,
                        metrics=METRICS_LIST,
                        df_train=df_train_max
                    )
                    for metric in METRICS_LIST:
                        cur_statistics[f'{metric}_test'] = [test_metrics.get(metric)]
                except Exception as e:
                    cur_statistics['error'] = [1]
                    cur_statistics['error_text'] = [f'EVAL_TEST_ERROR${str(e)}']
                write_dict(RES_FILENAME, cur_statistics)
            print(f"Done with model: {model_id}")
        except Exception as e:
            print(f"Error with model {row.get('model_id', idx)}: {e}")
