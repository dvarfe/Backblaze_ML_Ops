from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from disk_analyzer.utils.constants import EPOCHS, TIMES


class ClassifierArchitecture(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ClassifierArchitecture, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DLClassifier:
    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 1e-3, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = ClassifierArchitecture(input_dim, hidden_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self.is_fitted = False

    def fit(self, dataloader: DataLoader, epochs: int = EPOCHS):
        """Train or fine-tune the model on given data

        Args:
            dataloader (DataLoader): DataLoader with data
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self._model.train()
        for epoch in range(epochs):
            total_loss = 0
            for _, _, X, y, _ in dataloader:
                X = X.to(self.device).float()
                y = y.squeeze().to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self._model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        self.is_fitted = True

    def predict(self, dataloader: DataLoader, times: np.ndarray = TIMES) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get survival function

        Args:
            dataloader (DataLoader): Dataloader with data

        Returns:
            pd.DataFrame: DataFrame with survival function for each observation
        """

        self._model.eval()
        rows_pred = []
        rows_gt = []

        with torch.no_grad():
            for serial_numbers, time, X, y, real_durations in dataloader:
                X = X.cpu().numpy()
                for cur_serial_number, cur_time, line, cur_y, event_time in zip(serial_numbers, time, X, y, real_durations):
                    data_extended = torch.Tensor([list(line) + [time] for time in times]).to(self.device)
                    hazards = self._model(data_extended).squeeze().cpu().numpy()
                    cum_hazards = hazards.cumsum()
                    surv_f = np.exp(-cum_hazards)
                    row_pred = {
                        'serial_number': cur_serial_number,
                        'time': int(cur_time.cpu()),
                        **dict(zip(times, surv_f))
                    }
                    rows_pred.append(row_pred)

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
