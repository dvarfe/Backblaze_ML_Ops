from typing import List, Tuple

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np


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

        for _, _, X, y, _ in dataloader:
            X_np = X.numpy()
            y_np = np.ravel(y.numpy())

            if not self._is_fitted:
                self._model.partial_fit(X_np, y_np, classes=[0, 1])
                self._is_fitted = True
            else:
                self._model.partial_fit(X_np, y_np)

    def predict(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Predict class labels using the trained model.
        """
        rows = []

        for serial_numbers, time, X, y, real_durations in dataloader:
            X = X.numpy().reshape(-1, X.shape[-1])
            hazards = self._model.predict_proba(X)[:, 1]

            df_batch = pd.DataFrame({
                'serial_number': np.array(serial_numbers).flatten(),
                'time': np.array(time).flatten(),
                'pred': hazards.flatten()
            })

            df_batch['cum_hazard'] = df_batch.groupby(['serial_number', 'time'])['pred'].cumsum()
            df_batch['survival_f'] = np.exp(-df_batch['cum_hazard'])

            # Each group represents one prediction
            grouped = df_batch.groupby(['serial_number', 'time'])

            for (serial, t), group in grouped:
                surv_values = group['survival_f'].to_numpy()
                row = {
                    'serial_number': serial,
                    'time': t,
                    **dict(zip(dataloader.dataset.times, surv_values))
                }
                rows.append(row)

        df_surv = pd.DataFrame(rows)
        columns_order = ['serial_number', 'time'] + list(dataloader.dataset.times)
        df_surv = df_surv[columns_order]

        return df_surv
