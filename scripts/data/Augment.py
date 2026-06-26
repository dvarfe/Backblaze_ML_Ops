# Нужно приучить модель обращать внимание на время,
# а для этого нужно показать ей, что оно играет роль

import os
from glob import glob
import pandas as pd

OLD_DIR = 'Preprocessed/'
NEW_DIR = 'Augmented_more/'
NEW_SAMPLES = 20
TIME_DIST = 5

os.makedirs(NEW_DIR, exist_ok=True)
paths = glob(f'{OLD_DIR}*.csv')

for path in paths:
    df = pd.read_csv(path)
    filename = os.path.basename(path)

    terminal_mask = df['failure'] == 1
    terminal_df = df[terminal_mask].copy()

    augmented_dfs = [terminal_df]  # ATTENTION! PREVIOUSLY IT WAS df.copy()

    # Генерируем искусственные наблюдения
    for i in range(1, NEW_SAMPLES + 1):
        temp_df = terminal_df.copy()
        temp_df['max_lifetime'] = temp_df['max_lifetime'] - i * TIME_DIST

        valid_mask = temp_df['max_lifetime'] > temp_df['time']
        temp_df = temp_df[valid_mask]

        if not temp_df.empty:
            temp_df['failure'] = 0
            augmented_dfs.append(temp_df)

    augmented_df = pd.concat(augmented_dfs, ignore_index=True)
    augmented_df.to_csv(f'{NEW_DIR}{filename}', index=False)

    added = len(augmented_df) - len(df)
    print(f"File: {filename} | Original: {len(df)} | Added: {added} | Total: {len(augmented_df)}")
