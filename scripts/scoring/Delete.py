import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from disk_analyzer.models import DiskDataset
from disk_analyzer.research.config import DEFAULT_TIMES as TIMES
from disk_analyzer.research.aggregation.predictions_aggregator import PredictionsAggregator
from disk_analyzer.research.paths import artifacts_dir
import pickle

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EXP_NUM = 131
RES_FOLDER = str(artifacts_dir(EXP_NUM))
MODELS_FOLDER = os.path.join(RES_FOLDER, "models")
BATCH_SIZE = 1024
DATA_EXT = 'parquet'
DATA = f'Data/Preprocessed_new_data/20_train_preprocessed.{DATA_EXT}'

with open(os.path.join(MODELS_FOLDER, '0_CoxTILN.pkl'), 'rb') as f:
    model = pickle.load(f)

dl = DataLoader(dataset = DiskDataset('score', [DATA]),
        batch_size = BATCH_SIZE)

df_test = pd.read_csv(DATA) if DATA_EXT == 'csv' else pd.read_parquet(DATA) 
df_test = df_test[df_test['time'] != df_test['max_lifetime']]

AGG_MODE = 't_dist'
AGG_WEIGHT = 0.1
pred_agg = PredictionsAggregator(mode=AGG_MODE, weight=AGG_WEIGHT)
extended_times = pred_agg.get_extended_times(df_test, times = TIMES)
X_pred, X_gt = model.predict(dl, times=TIMES)