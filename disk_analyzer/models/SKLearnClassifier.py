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

        for X, y in dataloader:
            X_np = X.numpy()
            y_np = y.numpy()

            if not self._is_fitted:
                self._model.partial_fit(X_np, y_np, classes=[0, 1])
                self._is_fitted = True
            else:
                self._model.partial_fit(X_np, y_np)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Predict class labels using the trained model.
        """
        X_all = []
        for X, _ in dataloader:
            X_all.append(X.numpy())
        X_np = np.concatenate(X_all, axis=0)
        return self._model.predict(X_np)

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """
        Predict class probabilities using the trained model.
        """
        X_all = []
        for X, _ in dataloader:
            X_all.append(X.numpy())
        X_np = np.concatenate(X_all, axis=0)
        return self._model.predict_proba(X_np)
