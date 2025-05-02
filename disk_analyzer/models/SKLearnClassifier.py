from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from disk_analyzer.utils.constants import TIMES


class SKLClassifier():
    """Wrapper class for sklearn models that implement partial_fit method
    """

    def __init__(self, model):
        """_summary_

        Args:
            model: A scikit-learn model that implements `partial_fit`
        """
        self._model = model
        self._is_fitted = False

    def fit(self, dataloader: DataLoader):
        """
        Incrementally trains the model using batches from the dataloader.
        """

        for _, _, X, y, time_to_event in dataloader:
            X_np = X.numpy()
            X_np = np.concat([X, time_to_event.reshape(-1, 1)], axis=-1)
            y_np = np.ravel(y.numpy())

            if not self._is_fitted:
                self._model.partial_fit(X_np, y_np, classes=[0, 1])
                self._is_fitted = True
            else:
                self._model.partial_fit(X_np, y_np)

    def predict(self, dataloader: DataLoader, times: np.ndarray = TIMES) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get survival function

        Args:
            dataloader (DataLoader): Dataloader with data

        Returns:
            pd.DataFrame: DataFrame with survival function for each observation
        """
        # serials are contained separately, because they are strings
        pred_chunks = []
        pred_serials = []
        gt_chunks = []

        n_times = len(times)

        for serial_numbers, obs_times, X, y, real_durations in tqdm(dataloader):
            batch_size = X.size(0)
            serial_numbers = np.array(serial_numbers)
            X = np.array(X)

            # Expand each observation on times
            # Repeat each observation in batch len(times) times
            expanded_X = np.repeat(X.reshape(X.shape[0], -1, X.shape[1]), len(times), axis=1)
            expanded_times = np.repeat(times.reshape(1, -1, 1), batch_size, axis=0)
            data_extended = np.concat([expanded_X, expanded_times], axis=-1)

            hazards = self._model.predict_proba(data_extended.reshape(
                batch_size * len(times), -1))[:, 1]  # Flatten batches
            hazards = hazards.reshape(batch_size, n_times)  # Get vector of predictions for each batch
            surv_probs = np.exp(-hazards.cumsum(axis=1))

            pred_block = np.column_stack([
                obs_times,
                surv_probs
            ])

            pred_chunks.append(pred_block)
            pred_serials.append(serial_numbers)

            if (real_durations != -1).any():
                real_durations = real_durations
                y = y
                # Process if there are true lifetime values
                gt_block = np.column_stack([
                    obs_times,
                    real_durations - obs_times,
                    y
                ])
                gt_chunks.append(gt_block)

        pred_values = np.concatenate(pred_chunks, axis=0)
        serial_numbers_flat = np.concatenate(pred_serials)

        df_surv = pd.DataFrame(pred_values, columns=['time'] + times.tolist())
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')
        df_surv[times] = df_surv[times].astype('float32')

        if gt_chunks:
            gt_values = np.concatenate(gt_chunks, axis=0)
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({
                'serial_number': 'string',
                'time': 'int32',
                'duration': 'int32'
            })
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader, times=TIMES):
        df_survival, df_gt = self.predict(dataloader, times=times)
        return self.get_expected_time_by_predictions(df_survival, times), df_gt

    def get_expected_time_by_predictions(self, X_pred, times):
        X = X_pred
        survival_vec = X.drop(['serial_number', 'time'], axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
