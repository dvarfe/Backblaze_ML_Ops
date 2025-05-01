# Формат датафреймов, который использовался раньше при обучении 
# немного отличается от того, который используется сейчас, поэтому
# нужен этот скрипт, который конвертирует одно в другое

import os
from glob import glob
import pandas as pd
OLD_DIR = 'Data/Preprocessed/'
NEW_DIR = 'Data/Preprocessed_new/'

paths = glob(f'{OLD_DIR}*.csv')
os.makedirs(NEW_DIR)

for path in paths:
    df = pd.read_csv(path)
    filename = os.path.basename(path)
    df = df.rename(columns={'id': 'serial_number', 'event':'failure'})
    df['max_lifetime'] = df.groupby('serial_number')['time'].transform('max')
    df.to_csv(NEW_DIR + filename, index=False)