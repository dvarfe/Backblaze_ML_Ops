from typing import List, Tuple

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

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

        for _, _, X, y, _ in dataloader:
            X_np = X.numpy()
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

        rows_pred = []
        rows_gt = []

        for serial_numbers, time, X, y, event_times in dataloader:
            X = X.cpu().numpy()
            for cur_serial_number, cur_time, line, cur_y, event_time in zip(serial_numbers, time, X, y, event_times):
                data_extended = np.array([list(line) + [time] for time in times])
                hazards = self._model.predict_proba(data_extended)[:, 1]
                cum_hazards = hazards.cumsum()
                surv_f = np.exp(-cum_hazards)
                row = {
                    'serial_number': cur_serial_number,
                    'time': int(cur_time),
                    **dict(zip(times, surv_f))
                }
                rows_pred.append(row)

                if event_time != -1:
                    row_gt = {
                        'serial_number': cur_serial_number,
                        'time': int(cur_time.cpu()),
                        'duration': int((event_time - cur_time).cpu()),
                        'failure': bool(cur_y.cpu()),
                    }
                    rows_gt.append(row_gt)

        df_surv = pd.DataFrame(rows_pred)
        columns_order = ['serial_number', 'time'] + list(times)
        df_surv = df_surv[columns_order]

        df_gt = pd.DataFrame(rows_gt)

        return df_surv, df_gt

    def get_expected_time(self, dataloader, times=TIMES):
        df_survival, df_gt = self.predict(dataloader, times=times)
        return self.get_expected_time_by_predictions(df_survival, times), df_gt

    def get_expected_time_by_predictions(self, X_pred, times):
        X = X_pred
        survival_vec = X.drop(['serial_number', 'time'], axis='columns').values
        return np.trapezoid(y=survival_vec, x=times)
