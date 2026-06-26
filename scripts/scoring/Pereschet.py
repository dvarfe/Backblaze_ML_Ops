import pickle

import numpy as np
from torch.utils.data import DataLoader

from disk_analyzer.models.Cox import CoxTimeVaryingEstimator, CoxTimeInvariantSNFitter, CoxTimeInvariantLNFitter
# from BACKUP.Exp_56.code.SurvPredictor import SurvPredictor
from disk_analyzer.models.SurvPredictor import SurvPredictor
from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.stages.model_scoring import ModelScorer

PEN, L1 = 1, 1
LR = 5e-6
COX_MODELS = {'CoxTV': CoxTimeVaryingEstimator(penalizer=PEN, l1_ratio=L1),
              'CoxPH': CoxTimeInvariantSNFitter(penalizer=PEN, l1_ratio=L1)}

DATA_FOLDER = 'Preprocessed_new'
TRAIN_BATCHSIZE = 1024
TIMES = np.arange(0, 730)  # 729 - max duration in 2016, 2017
METRICS_LIST = ['ibs', 'ci']
SCORER = ModelScorer()


def sample_nth(df, n):
    df = df.sort_values(by=['serial_number', 'time'])
    df_sampled = df.groupby('serial_number').head(n).groupby('serial_number').tail(1)
    df_sampled = df_sampled.drop_duplicates().sort_values(
        by=['serial_number', 'time'])
    return df_sampled


def score_model(model, dl_score):
    X_pred, X_gt = model.predict(dl_score, TIMES)
    X_pred = sample_nth(X_pred, 10)
    X_gt = X_gt.loc[X_pred.index, :]
    return SCORER.get_metrics(model, X_pred, X_gt, TIMES, METRICS_LIST)


# dl_train = DataLoader(
#     dataset=DiskDataset('train', [f'{DATA_FOLDER}/1_train_preprocessed.parquet'],
#                         to_cens_time_list=[], to_term_time_list=[], cens_prob=-1),
#     batch_size=TRAIN_BATCHSIZE)

# dl_score = DataLoader(dataset=DiskDataset('score', [f'{DATA_FOLDER}/1_20_test_preprocessed.parquet'],
#                                           to_cens_time_list=[], to_term_time_list=[], cens_prob=-1),
#                       batch_size=TRAIN_BATCHSIZE)

# for model_name in COX_MODELS:
#     print(f'Обучение {model_name}')
#     model = COX_MODELS[model_name]
#     model.fit(dl_train)
#     metrics = score_model(model, dl_score)
#     print(f'Модель: {model_name}, {metrics}')

# sp = SurvPredictor(input_dim=28, hidden_dim=2048, epochs=100, lr=LR)
# sp.fit(dl_train, times=TRAIN_TIMES, val_dataloader=dl_score, early_stopping=True,
#        score_metric=SCORE_METRIC, patience=PATIENCE, min_delta=MIN_DELTA)

# metrics = score_model(sp, dl_score)
# print(f'Модель: sp, {metrics}')
# with open(f"sp_model.pkl", 'wb') as f:
#     pickle.dump(sp, f)

# Модель: sp, {'ci': 0.8088101416170685, 'ibs': 0.2014048042005427}
# Модель: CoxPH, {'ci': 0.8029552650326631, 'ibs': 0.142094318820147}
# Модель: CoxTV, {'ci': 0.8029766018304602, 'ibs': 0.14210913848341172}


# --------------------------------
dl_train = DataLoader(
    dataset=DiskDataset('train', [f'{DATA_FOLDER}/1_train_preprocessed.csv'],
                        to_cens_time_list=[], to_term_time_list=[], cens_prob=-1),
    batch_size=TRAIN_BATCHSIZE)

dl_score = DataLoader(dataset=DiskDataset('score', [f'{DATA_FOLDER}/1_25_test_preprocessed.csv'],
                                          to_cens_time_list=[], to_term_time_list=[], cens_prob=-1),
                      batch_size=TRAIN_BATCHSIZE)

# model = CoxTimeVaryingEstimator(penalizer=PEN, l1_ratio=L1)
# model.fit(dl_train)
# metrics = score_model(model, dl_score)
# print(metrics)

model = CoxTimeInvariantLNFitter(penalizer=PEN, l1_ratio=L1)
model.fit(dl_train)
metrics = score_model(model, dl_score)
print(metrics)

# with open("BACKUP/Exp_56/models/6_SP_model.pkl", 'rb') as f:
#     model = pickle.load(f)
# metrics = score_model(model, dl_score)
# print(metrics)

# with open("Artifacts/Exp_77/models/7_CoxTV_model.pkl", 'rb') as f:
#     model = pickle.load(f)
# metrics = score_model(model, dl_score)
# print(metrics)

# with open("Artifacts/Exp_72/models/7_CoxTV_model.pkl", 'rb') as f:
#     model = pickle.load(f)
# metrics = score_model(model, dl_score)
# print(metrics)
