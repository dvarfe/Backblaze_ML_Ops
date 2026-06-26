from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .ddh import DynamicDeepHit


class DDH(DynamicDeepHit):

    def __init__(self,
                 learning_rate=1e-3,
                 event_col="failure",
                 time_col='time',
                 id_col='serial_number',
                 **kwargs):

        super().__init__(
            **kwargs
        )
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.learning_rate = learning_rate

    def _batch_to_df(self, batch):
        serial_numbers, obs_times, X, y, durations = batch
        if isinstance(serial_numbers, (list, tuple)):
            serial_numbers = np.array(serial_numbers)
        elif torch.is_tensor(serial_numbers):
            serial_numbers = serial_numbers.cpu().numpy() if serial_numbers.is_cuda else serial_numbers.numpy()

        if torch.is_tensor(obs_times):
            obs_times = obs_times.cpu().numpy() if obs_times.is_cuda else obs_times.numpy()
        else:
            obs_times = np.array(obs_times)

        if torch.is_tensor(y):
            y = y.cpu().numpy() if y.is_cuda else y.numpy()
        else:
            y = np.array(y)
        y = y.astype(int)

        if torch.is_tensor(durations):
            durations = durations.cpu().numpy() if durations.is_cuda else durations.numpy()
        else:
            durations = np.array(durations)

        if torch.is_tensor(X):
            features = X.cpu().numpy() if X.is_cuda else X.numpy()
        else:
            features = np.array(X)

        # Создаем DataFrame
        df = pd.DataFrame(features)
        df[self.id_col] = serial_numbers.astype(str)
        df[self.event_col] = y
        df['duration'] = durations.astype(float)
        df[self.time_col] = obs_times.astype(int)

        return df

    def _df_to_ddh(self, df_in, agg_horizon=None, trunc_right=None):
        df = df_in.sort_values([self.id_col, self.time_col])

        if agg_horizon and trunc_right:
            df = df.groupby(self.id_col).head(trunc_right).groupby(self.id_col).tail(agg_horizon).copy()
        elif trunc_right:
            df = df.groupby(self.id_col).head(trunc_right).copy()
        elif agg_horizon:
            df = df.groupby(self.id_col).tailо(agg_horizon).copy()

        feature_cols = df.drop(columns=[self.event_col, 'duration', self.id_col, self.time_col]).columns

        x = []
        e = []
        t = []

        for id_val, group in df.groupby(self.id_col):
            # Извлекаем фичи
            x_group = group[feature_cols].values.astype(np.float64)
            x.append(x_group)

            # Извлекаем события
            e_group = group[self.event_col].values.astype(np.int64)
            e.append(e_group)

            # Извлекаем времена
            t_group = group['duration'].values.astype(np.float64)
            t.append(t_group)

        return x, t, e

    def fit(self,
            train_dataloader: DataLoader,
            times: np.ndarray,
            val_dataloader: Optional[DataLoader] = None,
            vsize: float = 0.15,
            iters: int = 10,
            batch_size: int = 100,
            optimizer: str = "Adam",
            random_state: int = 100):
        """
        Fit the DDH model to the data from train_dataloader.

        Args:
            train_dataloader: DataLoader yielding batches (X, t, e, serial_number, ...)
            times: Array of time points for prediction (used for discretization)
            val_dataloader: Optional DataLoader with validation data
            vsize: Fraction for validation split if val_dataloader is None
            Remaining args: Passed to DynamicDeepHit.fit()
        """
        if not train_dataloader:
            raise ValueError("train_dataloader cannot be None or empty")

        if times is None or len(times) == 0:
            raise ValueError("times array cannot be None or empty")

        try:
            dfs = []
            for batch in tqdm(train_dataloader, desc="Collecting data for DDH fit"):
                dfs.append(self._batch_to_df(batch))

            if not dfs:
                raise ValueError("No data collected from train_dataloader")

            df_all = pd.concat(dfs, ignore_index=True)

            if df_all.empty:
                raise ValueError("Collected DataFrame is empty")

            x_arr, t_arr, e_arr = self._df_to_ddh(df_all)

            super().fit(
                x_arr, t_arr, e_arr,
                vsize=vsize,
                iters=iters,
                learning_rate=self.learning_rate,
                batch_size=batch_size,
                optimizer=optimizer,
                random_state=random_state
            )
        except Exception as e:
            raise RuntimeError(f"Error during DDH fit: {str(e)}") from e

        return self

    def predict(self, dataloader: DataLoader, times: np.ndarray, risk: int = 1, bs: int = 100, all_step=False, agg_horizon=None, trunc_right=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions with the DDH model.

        Args:
            dataloader: DataLoader yielding batches (X, t, e, serial_number, ...)
            times: Array of time points to predict
            risk: Risk index (default 1)
            bs: Batch size for DDH prediction

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 
                - A DataFrame containing predicted survival functions for each observation.
                  Columns: ['serial_number', 'time', t1, t2, ..., tN]
                - A DataFrame containing ground truth durations and event indicators if available,
                  otherwise an empty DataFrame.
        """
        if not hasattr(self, 'fitted') or not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict. Call fit() first.")

        if not dataloader:
            raise ValueError("dataloader cannot be None or empty")

        if times is None or len(times) == 0:
            raise ValueError("times array cannot be None or empty")

        dfs = []

        for batch in tqdm(dataloader, desc="Collecting data for DDH predict"):
            df_batch = self._batch_to_df(batch)
            dfs.append(df_batch)

        df_all = pd.concat(dfs, ignore_index=True)

        x, t, e = self._df_to_ddh(df_all, agg_horizon=agg_horizon, trunc_right=trunc_right)
        serial_numbers = df_all[self.id_col].unique()
        # Для времени возьмём последнее наблюдение по каждому serial_number
        times_by_serial = df_all.groupby(self.id_col)['duration'].last().values

        surv_pred = self.predict_survival(
            x, times, risk=risk, all_step=all_step, bs=bs
        )

        df_pred = pd.DataFrame(surv_pred, columns=times.tolist())
        df_pred.insert(0, 'time', times_by_serial)
        df_pred.insert(0, 'serial_number', serial_numbers)

        # Для gt берем только последнее наблюдение для каждого serial_number
        if not df_all[self.event_col].isna().all() and not (df_all[self.event_col] == -1).all():
            df_gt_temp = df_all.sort_values([self.id_col, self.time_col]).groupby(self.id_col).last().reset_index()

            df_gt = pd.DataFrame({
                'serial_number': df_gt_temp[self.id_col].astype(str),
                'time': df_gt_temp[self.time_col].astype('int32'),
                'duration': df_gt_temp['duration'].astype('int32'),
                'failure': (df_gt_temp[self.event_col] == 1)
            })
        else:
            df_gt = pd.DataFrame()

        return df_pred, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes expected time to event given DataLoader, for each observation
        using trapezoidal rule.

        Args:
            dataloader: DataLoader yielding batches (X, t, e, serial_number, ...).
            times: time points array

        Returns:
            tuple[np.ndarray, pd.DataFrame]:
                - A numpy array of expected times to event for each observation
                - A DataFrame containing ground truth durations and event indicators if available,
                  otherwise an empty DataFrame.
        """
        df_pred, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_pred, times), df_gt

    def get_expected_time_by_predictions(self, df_pred: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """
        Calculates expected time to event based on predicted survival functions.

        Args:
            df_pred: DataFrame with columns ['serial_number', 'time', t1, ..., tN]
            times: Array of time points as used for prediction

        Returns:
            np.ndarray: Expected time for each observation
        """
        survival_vec = df_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapz(y=survival_vec, x=times)

    def count_parameters(self, trainable_only: bool = False) -> Dict[str, int]:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before counting parameters. Call fit() first.")

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for param in self.torch_model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params

        result = {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }

        if trainable_only:
            return trainable_params

        return result
