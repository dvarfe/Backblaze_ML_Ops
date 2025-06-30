# Программа, которая обучает модели, перебирает разные параметры и замеряет качество на тесте
import time
import pickle
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from disk_analyzer.models.DLClassifier import DLClassifier
from disk_analyzer.models.SKLearnClassifier import SKLClassifier
from disk_analyzer.models.Dataset import DiskDataset

from disk_analyzer.stages import ModelScorer


SCHEMA = {'train_samples' : [],
          'test_samples' : [],
            'method': [],
            'hidden_dim':[],
            'ci_train_same_size': [],
            'ci_train_max_size': [],
            'ci_test': [],
            'ibs_train_same_size': [],
            'ibs_train_max_size': [],
            'ibs_test': [],
            'ibs_bal_train_same_size': [],
            'ibs_bal_train_max_size': [],
            'ibs_bal_test': [],
            'train_time': [],
            'test_time': [],
            'error': [],
            'error_text': [],
            'model_id': []}


def write_dict(filename, dict_to_save):
    df = pd.DataFrame(dict_to_save)
    df.to_csv(filename, mode='a', header=False, index=False)

def create_res_file(filename):
    schema = SCHEMA
    df = pd.DataFrame(schema)
    df.to_csv(filename, index=False)

def save_model(model, model_id):
    with open(f'Artifacts\Exp_24\models/{model_id}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

RES_FILENAME = 'Artifacts\Exp_24\grid_search.csv'
TRAIN_GRID = [15, 20, 1, 2, 5, 10, 30, 40, 50]
TEST_GRID = [1, 25]
HIDDEN_DIM_GRID = [128, 512]
# METHODS = {'LogReg': SKLClassifier(SGDClassifier(loss='log_loss',  warm_start=True)), 'NN':DLClassifier(21)}
METHODS = ['NN']

TRAIN_BATCHSIZE = 10000
TIMES = np.arange(0, 730) #729 - max duration in 2016, 2017

if __name__ == "__main__":
    # Для скоринга мы берём неаугментированные данные
    if os.path.exists(RES_FILENAME):
        raise ValueError('Path exists!')
    create_res_file(RES_FILENAME)
    i = 0

    dl_score_max = DataLoader(
        dataset = DiskDataset('score', [f'Preprocessed/{max(TRAIN_GRID)}_train_preprocessed.csv']),
        batch_size = TRAIN_BATCHSIZE)
    
    scorer = ModelScorer()

    for train_samples in TRAIN_GRID:
        print(f'Начало обработки {train_samples} наблюдений')
        time_start = time.time()
        dl_train = DataLoader(
        dataset = DiskDataset('train', [f'Augmented/{train_samples}_train_preprocessed.csv']),
        batch_size = TRAIN_BATCHSIZE)
        for method in METHODS:
            for h_dim in HIDDEN_DIM_GRID:
                time_train_start = time.time()
                print(f'method={method}, hidden_dim={h_dim}')
                i += 1

                statistics = copy.deepcopy(SCHEMA)
                statistics['train_samples'] = train_samples
                statistics['test_samples'] = [None]
                statistics['method'] = [method]
                statistics['hidden_dim'] = h_dim
                statistics['ci_train_same_size'] = [None]
                statistics['ci_train_max_size'] = [None]
                statistics['ci_test'] = [None]
                statistics['ibs_train_same_size'] = [None]
                statistics['ibs_train_max_size'] = [None]
                statistics['ibs_test'] = [None]
                statistics['ibs_bal_train_same_size'] = [None]
                statistics['ibs_bal_train_max_size'] = [None]
                statistics['ibs_bal_test'] = [None]
                statistics['train_time'] = [None]
                statistics['test_time'] = [None]
                statistics['error'] = [0]
                statistics['error_text'] = ['']
                statistics['model_id'] = [str(i)+f'_{method}']
        
                model = DLClassifier(20, hidden_dim = h_dim, epochs=50)
                try:
                    model.fit(dl_train)
                except Exception as e:
                    statistics['error'] = 1
                    statistics['error_text'] = ['FIT_ERROR$' + str(e)]
                    write_dict('grid_search.csv', statistics)
                    continue
                print('Model is fit!')
                
                statistics['train_time'] = time.time() - time_train_start

                dl_train_score = DataLoader(
                                dataset = DiskDataset('score', [f'Preprocessed/{train_samples}_train_preprocessed.csv']),
                                batch_size = TRAIN_BATCHSIZE)
                df_train_predictions, df_train_gt = model.predict(dl_train_score, TIMES)
                statistics['ci_train_same_size'], \
                statistics['ibs_train_same_size'], \
                statistics['ibs_bal_train_same_size']  = scorer.get_ci_ibs_ibs_bal(model, df_train_predictions, df_train_gt, TIMES)

                df_train_max_predictions, df_train_max_gt = model.predict(dl_score_max, TIMES)
                statistics['ci_train_max_size'], \
                statistics['ibs_train_max_size'], \
                statistics['ibs_bal_train_max_size'] = scorer.get_ci_ibs_ibs_bal(model, df_train_max_predictions, df_train_max_gt, TIMES)

                for test_samples in TEST_GRID:
                    time_test_start = time.time()
                
                    cur_statistics = copy.deepcopy(statistics)
                    cur_statistics['test_samples'] = test_samples

                    dl_test_score = DataLoader(
                                dataset = DiskDataset('score', [f'Preprocessed/{train_samples}_{test_samples}_test_preprocessed.csv']),
                                batch_size = TRAIN_BATCHSIZE)
                    df_test_pred, df_test_pred_gt = model.predict(dl_test_score, TIMES) 
                    cur_statistics['ci_test'], \
                    cur_statistics['ibs_test'], \
                    cur_statistics['ibs_bal_test'] = scorer.get_ci_ibs_ibs_bal(model, df_test_pred, df_test_pred_gt, TIMES)
                    
                    cur_statistics['test_time'] = time.time() - time_test_start
                    print(f'ci_test: {cur_statistics["ci_test"]}, ibs_test: {cur_statistics["ibs_test"]}, test_samples: {test_samples}')

                    write_dict(RES_FILENAME, cur_statistics)
                    
                save_model(model, str(i)+f'_{method}')
    print(f'Обработка {train_samples} завершена за {time.time() - time_start} секунд')
                



    
