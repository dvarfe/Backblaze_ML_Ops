from typing import List, Tuple
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

        for _, _, X, y in dataloader:
            X_np = X.numpy()
            y_np = y.numpy()

            if not self._is_fitted:
                self._model.partial_fit(X_np, y_np, classes=[0, 1])
                self._is_fitted = True
            else:
                self._model.partial_fit(X_np, y_np)

    def predict(self, dataloader: DataLoader, threshold: float = 0.5) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Predict class labels using the trained model.
        """
        serial_numbers, times, probs = self.predict_proba(dataloader)
        return serial_numbers, times, (probs >= threshold).astype(int)

    def predict_proba(self, dataloader: DataLoader) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Predict class probabilities using the trained model.
        """
        probs = []
        serial_numbers = []
        times = []
        for serial_number, time, X, _ in dataloader:
            outputs = self._model.predict_proba(X)[:, 1]
            serial_numbers.extend(serial_number)
            times.extend(time)
            probs.extend(outputs)
        return serial_numbers, times, np.array(probs)
