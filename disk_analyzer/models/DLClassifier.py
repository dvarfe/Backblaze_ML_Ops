from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from disk_analyzer.utils.constants import EPOCHS


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
            for _, _, X, y in dataloader:
                X = X.to(self.device).float()
                y = y.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self._model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        self.is_fitted = True

    def predict_proba(self, dataloader: DataLoader) -> Tuple[List[str], List[int], np.ndarray]:
        """Predict probabilities

        Args:
            dataloader (DataLoader): Dataloader with data

        Returns:
            np.ndarray: Probabilities for each instance
        """
        self._model.eval()
        probs = []
        serial_numbers = []
        times = []
        with torch.no_grad():
            for serial_number, time, X, _ in dataloader:
                X = X.to(self.device).float()
                outputs = self._model(X)
                serial_numbers.extend(serial_number)
                times.extend(time)
                probs.extend(outputs.squeeze().cpu().numpy())

        return serial_numbers, times, np.array(probs)

    def predict(self, dataloader: DataLoader, threshold: float = 0.5) -> Tuple[List[str], List[int], np.ndarray]:
        """Return predictions based on the threshold

        Args:
            dataloader (DataLoader): Dataloader with data
            threshold (float, optional): Thereshold for prediction. Defaults to 0.5.

        Returns:
            np.ndarray: predictions
        """
        serial_numbers, times, probs = self.predict_proba(dataloader)
        return serial_numbers, times, (probs >= threshold).astype(int)
