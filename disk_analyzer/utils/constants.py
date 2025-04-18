from typing import Union
import os

from sklearn.linear_model import SGDClassifier  # type: ignore
import numpy as np

# Data Collector constants
BATCHSIZE = 500_000
COLLECTOR_CFG = './analyzer_cfg.json'
STORAGE_PATH = './Data/Data_collected'

# Data Analyzer constants
STATIC_STATS = [
    'data_size',
    'min_date',
    'max_date',
    'mean_lifetime',
    'max_lifetime',
    'na_rate',
    'truncated_rate',
    'survival_rate',
    'failure_rate',
    'double_failures',
    'mean_time_between_observ',
    'mean_observ_per_day'
]

STATIC_STATS_DESCRIPTION = {
    'data_size': 'number of observations in the data.',
    'min_date': 'date of the first observation.',
    'max_date': 'date of the last observation.',
    'mean_lifetime': 'mean lifetime of disks.',
    'max_lifetime': 'maximum lifetime of disks.',
    'na_rate': 'rate of missing data in each column.',
    'truncated_rate': 'rate of truncated disks.',
    'survival_rate': 'rate of survived disks.',
    'failure_rate': 'rate of failed disks.',
    'double_failures': 'rate of disks that have failed more than once.',
    'mean_time_between_observ': 'mean time between observations of a disk.',
    'mean_observ_per_day': 'mean number of observations per day.'
}


DYNAMIC_STATS = [
    'mean_lifetime',
    'max_lifetime',
    'na_rate',
    'survival_rate',
    'failure_rate',
    'mean_observ'
]

DYNAMIC_STATS_DESCRIPTION = {
    'mean_lifetime': 'mean lifetime of disks of disks alive at the day/month.',
    'max_lifetime': 'maximum lifetime of disks alive at the day/month.',
    'na_rate': 'rate of nans.',
    'survival_rate': 'rate of disks that have survived at the day/month.',
    'failure_rate': 'rate of disks that have failed at the day/month.',
    'mean_observ_per_day': 'mean number of observations.'
}

# Model Pipeline constants
MODEL_TYPES = Union[SGDClassifier]

# Data Preprocessor constants
PREPROCESSOR_STORAGE = './Data/preprocessed'
TEST_SIZE = 0.2
TRAIN_SAMPLES = 10
FEATURES_TO_REMOVE = ['smart_1_normalized', 'smart_2_normalized', 'smart_3_normalized',
                      'smart_4_normalized', 'smart_5_normalized', 'smart_7_normalized',
                      'smart_8_normalized', 'smart_9_normalized', 'smart_10_normalized',
                      'smart_11_normalized', 'smart_12_normalized', 'smart_13_normalized',
                      'smart_15_normalized', 'smart_183_normalized', 'smart_184_normalized',
                      'smart_187_normalized', 'smart_188_normalized', 'smart_189_normalized',
                      'smart_190_normalized', 'smart_191_normalized', 'smart_192_normalized',
                      'smart_193_normalized', 'smart_194_normalized', 'smart_195_normalized',
                      'smart_196_normalized', 'smart_197_normalized', 'smart_198_normalized',
                      'smart_199_normalized', 'smart_200_normalized', 'smart_201_normalized',
                      'smart_223_normalized', 'smart_225_normalized', 'smart_240_normalized',
                      'smart_241_normalized', 'smart_242_normalized', 'smart_250_normalized',
                      'smart_251_normalized', 'smart_252_normalized', 'smart_254_normalized',
                      'smart_255_normalized']

# Training constants
EPOCHS = 1
TRAIN_BATCHSIZE = 32

# Scoring constants
TIMES = np.arange(1, 400)

# Model vault
MODELS_VAULT = 'Models'
DEFAULT_MODEL_PATH = os.path.join(MODELS_VAULT, 'default.pkl')
