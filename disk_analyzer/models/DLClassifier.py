from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
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
        # self.criterion = nn.MSELoss()
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
            for _, _, X, y, time_to_event in dataloader:
                X = X.to(self.device).float()
                time_to_event = time_to_event.to(self.device).int().unsqueeze(1)
                X = torch.concat([X, time_to_event], dim=-1)
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
        self._model.eval()
        # serials are contained separately, because they are strings
        pred_chunks = []
        pred_serials = []
        gt_chunks = []

        with torch.no_grad():
            # Make tensor out of times
            times_tensor = torch.as_tensor(times, device=self.device, dtype=torch.float32)
            n_times = len(times)

            for serial_numbers, obs_times, X, y, real_durations in tqdm(dataloader):
                batch_size = X.size(0)
                serial_numbers = np.array(serial_numbers)

                X = X.to(self.device)
                obs_times = obs_times.to(self.device).int()

                # Expand each observation on times
                expanded_X = X.unsqueeze(1).expand(-1, n_times, -1)  # Repeat each observation in batch len(times) times
                expanded_times = times_tensor.reshape(1, -1, 1).expand(batch_size, -1, -1)
                data_extended = torch.cat([expanded_X, expanded_times], dim=-1)

                hazards = self._model(data_extended.view(batch_size * len(times), -1))  # Flatten batches
                hazards = hazards.view(batch_size, n_times)  # Get vector of predictions for each batch
                surv_probs = torch.exp(-hazards.cumsum(dim=1))

                pred_block = torch.column_stack([
                    obs_times,
                    surv_probs
                ])

                pred_chunks.append(pred_block)
                pred_serials.append(serial_numbers)

                if (real_durations != -1).any():
                    real_durations = real_durations.to(self.device)
                    y = y.to(self.device)
                    # Process if there are true lifetime values
                    gt_block = torch.column_stack([
                        obs_times,
                        real_durations - obs_times,
                        y
                    ])
                    gt_chunks.append(gt_block)

        pred_values = torch.concat(pred_chunks, dim=0).cpu().numpy()
        serial_numbers_flat = np.concatenate(pred_serials)

        df_surv = pd.DataFrame(pred_values, columns=['time'] + times.tolist())
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')
        df_surv[times] = df_surv[times].astype('float32')

        if gt_chunks:
            gt_values = torch.concat(gt_chunks, dim=0).cpu().numpy()
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
