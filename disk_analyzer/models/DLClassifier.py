from typing import Optional

import torch.nn as nn
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader


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
        self.model = ClassifierArchitecture(input_dim, hidden_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.is_fitted = False

    def fit(self, dataloader: DataLoader, epochs: int = 10):
        """Train or fine-tune the model on given data

        Args:
            dataloader (DataLoader): DataLoader with data
            epochs (int, optional): Number of epochs to train for. Defaults to 10.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X, y in dataloader:
                X = X.to(self.device).float()
                y = y.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

        self.is_fitted = True

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """Predict probabilities

        Args:
            dataloader (DataLoader): Dataloader with data

        Returns:
            np.ndarray: Probabilities for each instance
        """
        self.model.eval()
        probs = []
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(self.device).float()
                outputs = self.model(X)
                probs.extend(outputs.squeeze().cpu().numpy())

        return np.array(probs)

    def predict(self, dataloader: DataLoader, threshold: float = 0.5) -> np.ndarray:
        """Return predictions based on the threshold

        Args:
            dataloader (DataLoader): Dataloader with data
            threshold (float, optional): Thereshold for prediction. Defaults to 0.5.

        Returns:
            np.ndarray: predictions
        """
        probs = self.predict_proba(dataloader)
        return (probs >= threshold).astype(int)
