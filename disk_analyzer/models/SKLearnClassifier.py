from typing import Tuple, List
import time

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss

from ..utils.constants import TIMES, EPOCHS


class SKLClassifier:
    """
    A wrapper class for scikit-learn models that implement the `partial_fit` method.

    This class allows incremental training of a model using a DataLoader and
    provides methods to predict survival functions and expected time to event for given observations.
    """

    def __init__(self, model, epochs: int = EPOCHS):
        """
        Initializes the SKLClassifier with a scikit-learn model.

        Args:
            model: A scikit-learn model instance that implements partial_fit
        """
        self._model = model
        self.epochs = epochs
        self.loss: List[float] = []
        self.fit_times: List[float] = []

    def fit(self, dataloader: DataLoader):
        """
        Incrementally trains the model using batches from the provided DataLoader.

        Each batch is converted to numpy arrays of features and target labels.

        Args:
            dataloader (DataLoader): A DataLoader providing batches of data as tuples:
                (serial_numbers, obs_times, X, y, time_to_event)
        """
        for epoch in range(self.epochs):
            total_loss = 0
            start_fit_time = time.time()
            with tqdm(dataloader, unit='batch') as tepoch:
                for _, _, X, y, time_to_event in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    X_np = X.numpy()
                    X_np = np.concat([X, time_to_event.reshape(-1, 1)], axis=-1)
                    y_np = np.ravel(y.numpy())

                    self._model.partial_fit(X_np, y_np, classes=[0, 1])
                    total_loss += log_loss(y_np, self._model.predict_proba(X_np))
                    tepoch.set_postfix(loss=total_loss)
            fit_time = time.time() - start_fit_time
            self.fit_times.append(fit_time)
            self.loss.append(total_loss)

    def predict(self, dataloader: DataLoader, times: np.ndarray = TIMES) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predicts survival functions for observations from the dataloader.

         For each observation, this method predicts the survival probability
         over a predefined set of time points using the trained model.

         Args:
             dataloader (DataLoader): A DataLoader providing batches of data in the form:
                 (serial_numbers, obs_times, X, y, real_durations)
             times (np.ndarray, optional): Array of time points at which the survival function is evaluated.
                 Defaults to TIMES.

         Returns:
             Tuple[pd.DataFrame, pd.DataFrame]:
                 - A DataFrame containing predicted survival functions for each observation.
                   Columns: ['serial_number', 'time', t1, t2, ..., tN]
                 - A DataFrame containing ground truth durations and event indicators if available,
                   otherwise an empty DataFrame.
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

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray = TIMES) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes the expected time to event for observations in the dataloader
        based on predicted survival functions.

        Args:
            dataloader (DataLoader): A DataLoader providing data for prediction.
            times (np.ndarray, optional): Array of time points used for evaluating the survival function.
                Defaults to TIMES.

        Returns:
            Tuple[np.ndarray, pd.DataFrame]:
                - A numpy array of expected times to event for each observation.
                - A DataFrame containing ground truth durations and event indicators if available,
                  otherwise an empty DataFrame.
        """
        df_survival, df_gt = self.predict(dataloader, times=times)
        return self.get_expected_time_by_predictions(df_survival, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """
        Calculates expected time to event based on predicted survival functions.

        The expected time is computed as the area under the survival curve
        for each observation using trapezoidal rule.

        Args:
            X_pred (pd.DataFrame): DataFrame containing predicted survival functions.
                Columns: ['serial_number', 'time', t1, t2, ..., tN]
            times (np.ndarray): Array of time points corresponding to the survival functions.

        Returns:
            np.ndarray: A numpy array of expected times to event for each observation.
        """
        X = X_pred
        survival_vec = X.drop(['serial_number', 'time'], axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
