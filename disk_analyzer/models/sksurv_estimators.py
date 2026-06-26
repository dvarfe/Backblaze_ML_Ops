"""scikit-survival estimators (RSF, GBSA) integrated with DiskDataset DataLoader."""

from typing import Union

import numpy as np
import pandas as pd
import torch
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.util import Surv
from torch.utils.data import DataLoader
from tqdm import tqdm


class _SksurvSurvivalEstimator:
    """Shared fit/predict logic for scikit-survival tree models."""

    MODEL_CLASS = None

    def __init__(
        self,
        event_col: str = "failure",
        time_col: str = "time",
        id_col: str = "serial_number",
        random_state: int = 42,
        **model_params,
    ):
        if self.MODEL_CLASS is None:
            raise NotImplementedError("MODEL_CLASS must be set in subclass")
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.random_state = random_state
        self.model_params = model_params
        self.feature_cols = None
        self._model = None

    def _batch_to_df(self, batch, columns):
        serial_numbers, obs_times, X, y, durations = batch
        features = X.cpu().numpy() if torch.is_tensor(X) else np.array(X)
        df = pd.DataFrame(features, columns=columns)
        df[self.id_col] = np.array(serial_numbers)
        df[self.time_col] = np.array(obs_times)
        df[self.event_col] = np.array(y).astype(bool)
        df["duration"] = np.array(durations)
        return df

    def _feature_cols(self, dataloader):
        if hasattr(dataloader, "dataset"):
            return dataloader.dataset.feature_cols
        return dataloader.feature_cols

    def _collect_df(self, dataloader: DataLoader) -> pd.DataFrame:
        feature_cols = self._feature_cols(dataloader)
        dfs = []
        for batch in tqdm(dataloader, desc=f"Collecting data for {self.MODEL_CLASS.__name__}"):
            dfs.append(self._batch_to_df(batch, feature_cols))
        return pd.concat(dfs, ignore_index=True)

    def fit(self, train_dataloader: DataLoader):
        df_all = self._collect_df(train_dataloader)
        self.feature_cols = [
            c for c in df_all.columns
            if c not in {self.id_col, self.time_col, self.event_col, "duration"}
        ]

        X = df_all[self.feature_cols].to_numpy(dtype=np.float32)
        y = Surv.from_arrays(
            event=df_all[self.event_col].astype(bool).to_numpy(),
            time=df_all["duration"].to_numpy(dtype=np.float32),
        )

        params = {"random_state": self.random_state, **self.model_params}
        self._model = self.MODEL_CLASS(**params)
        self._model.fit(X, y)
        return self

    def _survival_at_times(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32)
        surv_fns = self._model.predict_survival_function(X.to_numpy(dtype=np.float32))
        rows = []
        for fn in surv_fns:
            t_min, t_max = fn.domain
            clipped = np.clip(times, t_min, t_max)
            row = fn(clipped).astype(np.float32)
            row = np.where(times > t_max, 0.0, row)
            row = np.where(times < t_min, 1.0, row)
            rows.append(row)
        return np.row_stack(rows)

    def predict(self, data: Union[DataLoader, pd.DataFrame], times: np.ndarray):
        if isinstance(data, pd.DataFrame):
            return self.predict_dataframe(data, times)
        return self.predict_dataloader(data, times)

    def predict_dataloader(self, dataloader: DataLoader, times: np.ndarray):
        df_all = self._collect_df(dataloader)
        return self._predict_df_all(df_all, times)

    def predict_dataframe(self, df: pd.DataFrame, times: np.ndarray):
        return self._predict_df_all(df.copy(), times)

    def _predict_df_all(self, df_all: pd.DataFrame, times: np.ndarray):
        X_feat = df_all[self.feature_cols]
        times = np.array(times)
        surv = self._survival_at_times(X_feat, times)

        pred_values = np.column_stack([df_all[self.time_col].values, surv])
        serial_numbers_flat = df_all[self.id_col].values
        columns = ["time"] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        df_surv.insert(0, "serial_number", serial_numbers_flat)
        df_surv["time"] = df_surv["time"].astype("int32")
        df_surv[times.tolist()] = df_surv[times.tolist()].astype("float32")

        if "duration" in df_all.columns:
            gt_values = np.column_stack([
                df_all[self.time_col].values,
                df_all["duration"].values,
                df_all[self.event_col].values,
            ])
            df_gt = pd.DataFrame(gt_values, columns=["time", "duration", "failure"])
            df_gt.insert(0, "serial_number", serial_numbers_flat)
            df_gt = df_gt.astype({"serial_number": "string", "time": "int32", "duration": "int32"})
            df_gt["failure"] = df_gt["failure"].astype(bool)
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray):
        df_surv, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(["serial_number", "time"], axis="columns").values
        return np.trapz(y=survival_vec, x=times)


class RandomSurvivalForestEstimator(_SksurvSurvivalEstimator):
    """Random Survival Forest (RSF) from scikit-survival."""

    MODEL_CLASS = RandomSurvivalForest


class GradientBoostingSurvivalEstimator(_SksurvSurvivalEstimator):
    """Gradient Boosting Survival Analysis (GBSA) from scikit-survival."""

    MODEL_CLASS = GradientBoostingSurvivalAnalysis
