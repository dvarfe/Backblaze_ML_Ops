# Программа, которая обучает модели, перебирает разные параметры и замеряет качество на тесте
import torch
import time
import pickle
import copy
import shutil
import os  # noqa
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # noqa

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from disk_analyzer.stages import ModelScorer
from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.models.DLClassifier import DLClassifier
from disk_analyzer.models.SurvPredictor import SurvPredictor
from disk_analyzer.models.Cox import CoxTimeVaryingEstimator
from disk_analyzer.models.Net import MAX_CLIP

np.random.seed(42)

METRICS_LIST = {'ci', 'ibs', 'ibs_bal', 'iauc'}


def make_schema(metrics_list):
    schema = {
        'train_samples': [],
        'test_samples': [],
        'method': [],
        'hidden_dim': [],
        'train_time': [],
        'test_time': [],
        'error': [],
        'error_text': [],
        'model_id': []
    }
    for metric in metrics_list:
        schema[f'{metric}_train_same_size'] = []
        schema[f'{metric}_train_max_size'] = []
        schema[f'{metric}_test'] = []
    return schema


SCHEMA = make_schema(METRICS_LIST)

EXP_NUM = 66
DATA_FOLDER = "Preprocessed_new"
BASE_RES_FOLDER = os.path.join("Artifacts", f"Exp_{EXP_NUM}")
RES_FILENAME = os.path.join(BASE_RES_FOLDER, "grid_search.csv")
MODELS_FOLDER = os.path.join(BASE_RES_FOLDER, "models")
LOG_DIR = os.path.join(BASE_RES_FOLDER, "logs")
TRAIN_GRID = [1, 2, 5, 10, 15, 20, 30, 40, 50]
TEST_GRID = [1, 10]
HIDDEN_DIM_GRID = [2048]
TO_CENS_SHIFT = []  # range(1, 200, 25)
TO_TERM_SHIFT = []  # range(1, 200, 25)
CENS_PROB = -1
METHODS = ['Cox']
EPOCHS = 100
LR = 5e-6
EARLY_STOPPING = True
SCORE_METRIC = 'ibs'
PATIENCE = 10
MIN_DELTA = 0.001


TRAIN_BATCHSIZE = 512
SCORE_BATCHSIZE = 512
TIMES = np.arange(0, 730)  # 729 - max duration in 2016, 2017
TRAIN_TIMES = TIMES[0::10]


HPARAMS = {
    'exp_num': EXP_NUM,
    'test_grid': str(TEST_GRID),
    'hidden_dim_grid': str(HIDDEN_DIM_GRID),
    'to_cens_shift': str(TO_CENS_SHIFT),
    'to_term_shift': str(TO_TERM_SHIFT),
    'cens_prob': CENS_PROB,
    'epochs': EPOCHS,
    'lr': LR,
    'early_stopping': EARLY_STOPPING,
    'patience': PATIENCE,
    'min_delta': MIN_DELTA,
    "max_clip": MAX_CLIP,
    'train_bs': TRAIN_BATCHSIZE,
    'score_bs': SCORE_BATCHSIZE,
    'times': f"{TIMES.min()} - {TIMES.max()}",
    'train_times_step': 10,
    'methods': str(METHODS)
}


FILES_TO_SAVE = ["Experiments.py",
                 "disk_analyzer/models/Net.py",
                 "disk_analyzer/models/Dataset.py",
                 "disk_analyzer/models/SurvPredictor.py",
                 ]


def write_dict(filename, dict_to_save):
    df = pd.DataFrame(dict_to_save)
    df.to_csv(filename, mode='a', header=False, index=False)


def create_res_file(filename):
    schema = SCHEMA
    df = pd.DataFrame(schema)
    df.to_csv(filename, index=False)


def create_description_file():
    desc_path = os.path.join(BASE_RES_FOLDER, "Description.txt")
    os.mknod(desc_path)
    with open(desc_path, "w") as f:
        for key in HPARAMS:
            f.write(f"{key} = {HPARAMS[key]}\n")


def save_model(model, model_id):
    with open(os.path.join(MODELS_FOLDER, f"{model_id}_model.pkl"), 'wb') as f:
        pickle.dump(model, f)


def copy_code():
    code_dir = os.path.join(BASE_RES_FOLDER, "code")
    os.mkdir(code_dir)
    for file in FILES_TO_SAVE:
        shutil.copy(file, code_dir)


def init_experiments_folder():
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
    if os.path.exists(RES_FILENAME):
        raise ValueError('Path exists!')
    create_res_file(RES_FILENAME)
    create_description_file()
    copy_code()


if __name__ == "__main__":

    init_experiments_folder()
    i = 0

    dl_score_max = DataLoader(
        dataset=DiskDataset('score', [f'{DATA_FOLDER}/{max(TRAIN_GRID)}_train_preprocessed.csv']),
        batch_size=SCORE_BATCHSIZE,
    )

    scorer = ModelScorer()
    df_train_max = pd.read_csv(f'{DATA_FOLDER}/{max(TRAIN_GRID)}_train_preprocessed.csv')
    df_train_max['duration'] = df_train_max['max_lifetime'] - df_train_max['time']
    df_train_max = df_train_max[['duration', 'failure', 'time']]
    for train_samples in TRAIN_GRID:
        print(f'Начало обработки {train_samples} наблюдений')
        time_start = time.time()
        dl_train = DataLoader(
            dataset=DiskDataset('train', [f'{DATA_FOLDER}/{train_samples}_train_preprocessed.csv'],
                                to_cens_time_list=TO_CENS_SHIFT, to_term_time_list=TO_TERM_SHIFT, cens_prob=CENS_PROB),
            batch_size=TRAIN_BATCHSIZE)

        # Open Dataloader for validation during train
        dl_val = DataLoader(
            dataset=DiskDataset('score', [f'{DATA_FOLDER}/{train_samples}_1_test_preprocessed.csv'],
                                to_cens_time_list=TO_CENS_SHIFT, to_term_time_list=TO_TERM_SHIFT, cens_prob=CENS_PROB),
            batch_size=TRAIN_BATCHSIZE)
        for method in METHODS:
            for h_dim in HIDDEN_DIM_GRID:
                time_train_start = time.time()
                print(f'method={method}, hidden_dim={h_dim}')
                cur_run = f"{method}_{h_dim}_{train_samples}"
                cur_log_dir = os.path.join(LOG_DIR, cur_run)
                writer = SummaryWriter(cur_log_dir)
                i += 1

                statistics = copy.deepcopy(SCHEMA)
                statistics['train_samples'] = [train_samples]
                statistics['test_samples'] = [None]
                statistics['method'] = [method]
                statistics['hidden_dim'] = [h_dim]
                for metric in METRICS_LIST:
                    statistics[f'{metric}_train_same_size'] = [None]
                    statistics[f'{metric}_train_max_size'] = [None]
                    statistics[f'{metric}_test'] = [None]
                statistics['train_time'] = [None]
                statistics['test_time'] = [None]
                statistics['error'] = [0]
                statistics['error_text'] = ['']
                statistics['model_id'] = [str(i)+f'_{method}']

                if method == "NN":
                    model = DLClassifier(28, hidden_dim=h_dim, epochs=EPOCHS, lr=LR)
                elif method == "SP":
                    model = SurvPredictor(28, hidden_dim=h_dim, epochs=EPOCHS, lr=LR)
                elif method == "Cox":
                    model = CoxTimeVaryingEstimator(penalizer=0.01, l1_ratio=0.1)
                # model = SKLClassifier(SGDClassifier(loss='log_loss',  warm_start=True))
                try:
                    if method == "NN":
                        model.fit(dl_train, writer)
                    elif method == "SP":
                        model.fit(dl_train, times=TRAIN_TIMES, val_dataloader=dl_val, early_stopping=EARLY_STOPPING,
                                  score_metric=SCORE_METRIC, patience=PATIENCE, min_delta=MIN_DELTA, writer=writer)
                    elif method == "Cox":
                        model.fit(dl_train)
                except Exception as e:
                    statistics['error'] = 1
                    statistics['error_text'] = ['FIT_ERROR$' + str(e)]
                    write_dict(RES_FILENAME, statistics)
                    continue
                print('Model is fit!')

                statistics['train_time'] = time.time() - time_train_start

                dl_train_score = DataLoader(
                    dataset=DiskDataset('score', [f'{DATA_FOLDER}/{train_samples}_train_preprocessed.csv']),
                    batch_size=SCORE_BATCHSIZE)
                df_train_predictions, df_train_gt = model.predict(dl_train_score, TIMES)
                train_metrics = scorer.get_metrics(
                    model, df_train_predictions, df_train_gt, TIMES,
                    metrics=METRICS_LIST,
                    df_train=df_train_max
                )
                for metric in METRICS_LIST:
                    statistics[f'{metric}_train_same_size'] = [train_metrics.get(metric)]

                df_train_max_predictions, df_train_max_gt = model.predict(dl_score_max, TIMES)
                train_max_metrics = scorer.get_metrics(
                    model, df_train_max_predictions, df_train_max_gt, TIMES,
                    metrics=METRICS_LIST,
                    df_train=df_train_max
                )
                for metric in METRICS_LIST:
                    statistics[f'{metric}_train_max_size'] = [train_max_metrics.get(metric)]

                HPARAMS['n_train'] = train_samples

                metrics = {}
                for metric in METRICS_LIST:
                    metrics[f'{metric}/train_same'] = statistics.get(f'{metric}_train_same_size', [None])[0]
                    metrics[f'{metric}/train_max'] = statistics.get(f'{metric}_train_max_size', [None])[0]
                metrics['time/train'] = statistics['train_time']

                for test_samples in TEST_GRID:
                    time_test_start = time.time()

                    cur_statistics = copy.deepcopy(statistics)
                    cur_statistics['test_samples'] = test_samples

                    dl_test_score = DataLoader(
                        dataset=DiskDataset(
                            'score', [f'{DATA_FOLDER}/{train_samples}_{test_samples}_test_preprocessed.csv']),
                        batch_size=SCORE_BATCHSIZE)
                    df_test_pred, df_test_pred_gt = model.predict(dl_test_score, TIMES)
                    test_metrics = scorer.get_metrics(
                        model, df_test_pred, df_test_pred_gt, TIMES,
                        metrics=METRICS_LIST,
                        df_train=df_train_max
                    )
                    for metric in METRICS_LIST:
                        cur_statistics[f'{metric}_test'] = [test_metrics.get(metric)]

                    cur_statistics['test_time'] = time.time() - time_test_start
                    # Print all metrics for train and test
                    print(f"Train metrics for {train_samples} samples:")
                    for metric in METRICS_LIST:
                        print(
                            f"  {metric}_train_same_size: {cur_statistics.get(f'{metric}_train_same_size', [None])[0]}")
                    print(f"Test metrics for {test_samples} samples:")
                    for metric in METRICS_LIST:
                        print(f"  {metric}_test: {cur_statistics.get(f'{metric}_test', [None])[0]}")

                    # Update tensorboard metrics
                    for metric in METRICS_LIST:
                        metrics[f'{metric}/train_same'] = cur_statistics.get(f'{metric}_train_same_size', [None])[0]
                        metrics[f'{metric}/train_max'] = cur_statistics.get(f'{metric}_train_max_size', [None])[0]
                        metrics[f'{metric}/test_{test_samples}'] = cur_statistics.get(f'{metric}_test', [None])[0]

                    write_dict(RES_FILENAME, cur_statistics)

                writer.add_hparams(HPARAMS, metrics, run_name=os.path.dirname(
                    os.path.realpath(__file__)) + os.sep + cur_log_dir)
                save_model(model, str(i)+f'_{method}')
        print(f'Обработка {train_samples} завершена за {time.time() - time_start} секунд')
