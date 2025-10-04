import pandas as pd
import numpy as np
import torch
from lifelines import CoxTimeVaryingFitter
from torch.utils.data import DataLoader
from tqdm import tqdm


class CoxTimeVaryingEstimator(CoxTimeVaryingFitter):
    """
    CoxTimeVaryingEstimator: полностью совместим с новой инфраструктурой.
    Формирует корректные интервалы start/stop для TV-Cox на основе time и duration.
    """

    def __init__(self, penalizer=0.0, l1_ratio=0.0, event_col="failure", time_col='time', id_col='serial_number', device=None):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None

    def _batch_to_df(self, batch):
        serial_numbers, obs_times, X, y, durations = batch
        serial_numbers = np.array(serial_numbers)
        obs_times = np.array(obs_times)
        y = np.array(y).astype(int)
        durations = np.array(durations)
        features = X.cpu().numpy() if torch.is_tensor(X) else np.array(X)
        df = pd.DataFrame(features)
        df[self.id_col] = serial_numbers
        df[self.time_col] = obs_times
        df[self.event_col] = y # здесь можно поставить нарезкозависимое преобразование
        df['duration'] = durations 
        return df

    def fit(self, train_dataloader: DataLoader):
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_tv = self._to_start_stop(df_all)
        super().fit(df_tv, id_col=self.id_col, start_col='start', stop_col='stop', event_col=self.event_col)
        self.feature_cols = [c for c in df_tv.columns if c not in [
            self.id_col, 'start', 'stop', self.event_col, 'duration']]
        return self

    def _to_start_stop(self, df):
        """
        Формирует DataFrame для CoxTimeVaryingFitter:
        - start = time - min_time_in_group
        - stop = start следующего наблюдения, для последнего: start + duration
        - event_col для всех кроме последнего: из следующей строки, для последнего — текущее значение
        """
        df = df.sort_values([self.id_col, self.time_col]).copy()
        min_times = df.groupby(self.id_col)[self.time_col].transform('min')
        df['start'] = df[self.time_col] - min_times
        df['stop'] = df.groupby(self.id_col)['start'].shift(-1)
        last_mask = df['stop'].isna()
        df.loc[last_mask, 'stop'] = df.loc[last_mask, 'start'] + df.loc[last_mask, 'duration']
        df.loc[~last_mask, self.event_col] = 0 # Все наблюдения, кроме последнего цензурированые
        df = df.drop(columns=['duration'])
        return df

    def _get_survival_function(self, X_features, times):
        baseline_surv = self.baseline_survival_
        fill_indices = baseline_surv.index.searchsorted(times, side='right') - 1
        fill_indices = np.clip(fill_indices, 0, len(baseline_surv) - 1)
        baseline_surv_interp = baseline_surv.iloc[fill_indices, 0].values
        partial_haz = self.predict_partial_hazard(X_features).values
        surv = baseline_surv_interp[None, :] ** partial_haz[:, None]
        return surv

    def predict(self, dataloader: DataLoader, times: np.ndarray):
        pred_chunks = []
        pred_serials = []
        gt_chunks = []

        times = np.array(times)
        for batch in tqdm(dataloader, desc="Cox prediction"):
            batch_df = self._batch_to_df(batch)
            X_feat = batch_df[self.feature_cols]
            surv = self._get_survival_function(X_feat, times)
            batch_pred = np.column_stack([batch_df[self.time_col].values, surv])
            pred_chunks.append(batch_pred)
            pred_serials.append(batch_df[self.id_col].values)

            if 'duration' in batch_df:
                gt_block = np.column_stack([
                    batch_df[self.time_col].values,
                    batch_df['duration'].values,
                    batch_df[self.event_col].values
                ])
                gt_chunks.append(gt_block)

        pred_values = np.vstack(pred_chunks)
        serial_numbers_flat = np.concatenate(pred_serials)
        columns = ['time'] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')

        if gt_chunks:
            gt_values = np.vstack(gt_chunks)
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({'serial_number': 'string', 'time': 'int32', 'duration': 'int32'})
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray):
        df_surv, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapz(y=survival_vec, x=times)
