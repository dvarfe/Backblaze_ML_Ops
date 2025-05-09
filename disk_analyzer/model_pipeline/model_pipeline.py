import os
import shutil
from typing import Optional, Dict, List, Union

import pandas as pd
from sklearn.linear_model import SGDClassifier  # type: ignore
from torch.utils.data import DataLoader

from ..utils.constants import PREPROCESSOR_STORAGE, BATCHSIZE, TRAIN_BATCHSIZE, TIMES, MODELS_VC
from ..models import DLClassifier, SKLClassifier, DiskDataset
from ..stages import TrainTestSplitter, DataPreprocessor, ModelScorer, ModelVManager


class ModelPipeline:
    """Class that incorporates all pipeline logic
        Consists of the following stages:
            1. Data Preprocessor: split data into train and test, preprocess data
            2. Model: train and predict
            3. Scoring: get metrics
    """

    def __init__(self,
                 data_paths: Optional[List[str]] = None,
                 train_test_splitter: Optional[TrainTestSplitter] = None,
                 data_preprocessor: Optional[DataPreprocessor] = None,
                 model_scorer: Optional[ModelScorer] = None,
                 models_version_manager: Optional[ModelVManager] = None,
                 batchsize: int = BATCHSIZE,
                 prep_storage_path: str = PREPROCESSOR_STORAGE):

        self.batchsize = batchsize
        self.prep_storage_path = prep_storage_path
        self.data_paths = data_paths
        self._train_test_splitter = train_test_splitter or TrainTestSplitter()
        self._data_preprocessor = data_preprocessor or DataPreprocessor(storage_paths=data_paths)
        self._model_scorer = model_scorer or ModelScorer()
        self._model_version_manager = models_version_manager or ModelVManager()
        self._model: Union[DLClassifier, SKLClassifier]

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])

    def preprocess(self, data_paths: List[str], mode: str = 'train') -> str:
        """Preprocess data

        Args:
            data_paths (pd.DataFrame): Paths to data files.
            mode (str, optional): Whether to preprocess train/test/tune data. Defaults to 'train'.
        """
        # Split data into train and test
        train_id, test_id = self._train_test_splitter.train_test_split(data_paths)

        # Open data and pass it to preprocessor
        df = self.open_data(data_paths)
        df_train = df[df['serial_number'].isin(train_id)]
        df_test = df[~df['serial_number'].isin(test_id)]

        if mode == 'train':

            df_train, df_test = self._data_preprocessor.fit_transform(
                df_train), self._data_preprocessor.transform(df_test)

            self.features_num = df_train.shape[1] - 2

            print('Save Preprocessed Data')

            for sample_name, df in zip(['train', 'test'], [df_train, df_test]):
                sample_dir = os.path.join(self.prep_storage_path, sample_name)

                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)

                os.makedirs(sample_dir, exist_ok=True)

                df.sort_values(by=['serial_number', 'time'], inplace=True)

                total_rows = df.shape[0]
                for i in range((total_rows + self.batchsize - 1) // self.batchsize):
                    batch = df.iloc[i*self.batchsize: (i+1)*self.batchsize]
                    batch.to_csv(os.path.join(sample_dir, f'{i}_preprocessed.csv'), index=False)
        elif mode == 'test' or mode == 'tune':
            df_train, df_test = self._data_preprocessor.fit_transform(
                df_train), self._data_preprocessor.transform(df_test)

            self.features_num = df_train.shape[1] + 1

            print('Save Preprocessed Data')

            for sample_name, df in zip(['train', 'test'], [df_train, df_test]):
                sample_dir = os.path.join(self.prep_storage_path, sample_name)

                df.sort_values(by=['serial_number', 'time'], inplace=True)

                total_rows = df.shape[0]
                for i in range((total_rows + self.batchsize - 1) // self.batchsize):
                    batch = df.iloc[i*self.batchsize: (i+1)*self.batchsize]
                    batch.to_csv(os.path.join(sample_dir, f'{i}_preprocessed.csv'), index=False)
        return sample_dir

    def set_model(self, model_name: str, learn_params: Dict, model_params: Dict):
        """Set model

        Args:
            model : Model
            interface (str, optional): Sklearn or torch. Defaults to 'sklearn'.
        """
        if model_name == 'logistic_regression':
            self._model = SKLClassifier(SGDClassifier(warm_start=True, loss='log_loss', **model_params), **learn_params)
        elif model_name == 'NN':
            if 'input_dim' not in model_params and self.features_num is None:
                raise ValueError('input_dim must be specified')
            elif 'input_dim' in model_params:
                self._model = DLClassifier(**learn_params, **model_params)
            else:
                self._model = DLClassifier(input_dim=self.features_num, **learn_params, **model_params)
        else:
            raise ValueError('Invalid mode. Only "logistic_regression" and "NN" are supported')
        self.model_name = model_name

    def fit(self, paths: List[str]):
        """Fit model
        """
        if self._model is None:
            raise ValueError("Model not set")
        ds = DiskDataset('train', paths)

        dl = DataLoader(ds, batch_size=TRAIN_BATCHSIZE)

        self._model.fit(dl)
        self._model_version_manager.save_model(self)

    def predict(self, paths: List[str], mode: str = 'score', times=TIMES):
        """Return predictions
        """

        if self._model is None:
            raise ValueError("Model not fitted")

        ds = DiskDataset(mode, paths)

        dl = DataLoader(ds, batch_size=TRAIN_BATCHSIZE)

        return self._model.predict(dl, times)

    def score_model(self, paths: List[str], times=TIMES):
        """Return predictions
        """

        if self._model is None:
            raise ValueError("Model not fitted")

        ds = DiskDataset('score', paths)

        dl = DataLoader(ds, batch_size=TRAIN_BATCHSIZE)

        df_pred, df_gt = self._model.predict(dl, times)

        return self._model_scorer.get_ci_and_ibs(self._model, df_pred, df_gt, times)

    def save_best_model(self, metric: str, path: str):
        self._model_version_manager.save_best_model(metric, path)
