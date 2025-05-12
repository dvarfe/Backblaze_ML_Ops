import os
import glob
import pickle
from typing import Dict, Optional, Tuple, List
import json

import pandas as pd
from numpy.typing import NDArray
from numpy import int_

from ..stages import DataCollector, DataStats
from ..models import DLClassifier
from ..model_pipeline import ModelPipeline
from ..utils.constants import STORAGE_PATH, TIMES, DEFAULT_MODEL_PATH, PREPROCESSOR_STORAGE, REPORT_PATH


class Controller:
    """Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        """Initialize the Controller class.

        Sets up the model pipeline and paths attributes for data processing and analysis.
        Initially sets model_pipeline to None and creates an empty list for paths.
        """
        self.model_pipeline = None
        self.paths = []

    def collect_data(self, *args, **kwargs):
        """Collect data using the DataCollector.

        Args:
            *args: Variable positional arguments to be passed to DataCollector.
            **kwargs: Variable keyword arguments to be passed to DataCollector.
        """
        DataCollector(*args, **kwargs).collect_data()

    def rebatch(self, new_batchsize: int):
        """Rebatch data with a new batch size.

        Args:
            new_batchsize (int): The new batch size to use.
        """
        DataCollector(paths=[], batchsize=new_batchsize,
                      cfgpath='').collect_data()

    def get_data_statistics(self,  storage_path: str, static_stats: List[str], dynamic_stats: List[str], dynamic_freq: str, figpath: str) -> Tuple[dict, str]:
        """Returns statistics of the collected data.

        Args:
            storage_path (str): Path to the storage directory.
            static_stats (List[str]): Static stats to collect
            dynamic_stats (List[str]): Dynamic stats to collect
            dynamic_freq (str): Frequency of dynamic stats(daily or monthly)
            figpath (str): Path to save figures

        Returns:
            Tuple[dict, str]: Static stats and figures path
        """
        data_stats = DataStats(
            static_stats, dynamic_stats, dynamic_freq, figpath)
        paths = glob.glob(os.path.join(storage_path, '*.csv'))
        df = self.open_data(paths)
        stats = data_stats.calculate_stats(df)
        return stats

    def preprocess_data(self, storage_path: str = STORAGE_PATH, preprocessed_path: str = PREPROCESSOR_STORAGE, model_mode: str = 'train'):
        """Preprocess data for model training or inference.

        Finds all CSV files in the specified storage path and initializes
        a ModelPipeline for data preprocessing.

        Args:
            storage_path (str, optional): Path to the directory containing data files.
                                          Defaults to STORAGE_PATH.
            preprocessed_path (str, optional): Path to the directory to save preprocessed data.
                                               Defaults to PREPROCESSOR_STORAGE.
            model_mode (str, optional): Mode of preprocessing, either 'train' or 'inference'.
                                        Defaults to 'train'.
        """
        paths = glob.glob(os.path.join(storage_path, '*.csv'))
        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline(data_paths=paths, prep_storage_path=preprocessed_path)
        self.model_pipeline.preprocess(data_paths=paths, mode=model_mode)

    def update_preprocessed(self, new_dir: str):
        """Update preprocessed data with new data.

        Args:
            new_dir (str): Path to the directory containing new batched data.

        Raises:
            ValueError: If the model is not loaded and the default model is not found.
        """
        paths = glob.glob(os.path.join(new_dir, '*.csv'))
        if self.model_pipeline is None:
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.model_pipeline = self.load_model(DEFAULT_MODEL_PATH)
            else:
                raise ValueError('Model not loaded and default model not found')
        self.model_pipeline.preprocess(data_paths=paths, mode='tune')

    def fit(self, model_name: str, cfg: str, preprocessed_path: str):
        """Train a model on preprocessed data.
        Supports two model types: logistic regression (sklearn) and Neural Network (torch).
        Initializes the model and fits it on preprocessed training batches.

        Args:
            model_name (str): Name of the model to train. Supports 'logistic_regression' or 'NN'.
            cfg (str): Path to config file with model parameters.
            preprocessed_path (str): Path to preprocessed training data.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        model_params = {}
        learn_params = {}
        if os.path.exists(cfg):
            config = json.load(open(cfg, 'r'))
            if 'learn_params' in config:
                learn_params = config['learn_params']
            if 'model_params' in config:
                model_params = config['model_params']
        else:
            model_params = {}

        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline(prep_storage_path=preprocessed_path)
        self.model_pipeline.set_model(model_name, learn_params=learn_params, model_params=model_params)
        batches = glob.glob(os.path.join(preprocessed_path, 'train', '*.csv'))
        self.model_pipeline.fit(batches)

    def fine_tune(self, preprocessed_path: str):
        """Fine-tune a model on preprocessed data."""
        if self.model_pipeline is None:
            raise ValueError('You must load model first!')
        batches = glob.glob(os.path.join(preprocessed_path, 'train', '*.csv'))
        self.model_pipeline.fit(batches)

    def predict(self, path: str):
        """Model inference on a single file. The results are saved in the 'Predictions' folder.

        Args:
            path (str): Input file
        """
        predictions_path = 'Predictions/'
        df_pred, _ = self.model_pipeline.predict([path])
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        df_pred.to_csv(f'{predictions_path}/prediction.csv', index=False)
        print(f'Predictions saved to {os.path.join(predictions_path, "predictions.csv")}')

    def score_model(self, paths: Optional[List[str]] = None, times: NDArray[int_] = TIMES):
        """Score model on test data or on specified in paths files.

        Args:
            paths (Optional[List[str]], optional): List of test filepaths. If None score on test data. Defaults to None.
            times (NDArray[int\_], optional): Array of time points to use for scoring. Defaults to TIMES.
        """
        if paths is None:
            score_paths = glob.glob(os.path.join(self.model_pipeline.prep_storage_path, 'test', '*.csv'))
        else:
            score_paths = paths
        return self.model_pipeline.score_model(score_paths, times)

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Combine multiple CSV files into a single pandas DataFrame.

        Args:
            paths (List[str]): List of file paths to CSV files.

        Returns:
            pd.DataFrame: A consolidated DataFrame containing data from all input files.
        """
        return pd.concat([pd.read_csv(path) for path in paths])

    def save_model(self, path: str):
        """Save trained model into a file.

        Args:Dict[str, List[float]]
            path (str): Path to save the model file.
        """
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            pickle.dump(self.model_pipeline, f)

    def load_model(self, path: str):
        """Load trained model from a file.

        Args:
            path (str): Path to the model file.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        self.model_pipeline = model

    def save_best_model(self, metric: str, path: str):
        """Save the best model based on the specified metric (ci or ibs).

        Args:
            metric (str): The metric to use for selecting the best model ('ci' or 'ibs').
            path (str): Path to save the best model file.
        """
        self.model_pipeline.save_best_model(metric, path)

    def get_model_stats(self) -> Dict[str, List[float]]:
        """Get statistics of the model.

        Returns:
            Dict[str, List[float]]: Dictionary with model statistics.
        """
        return self.model_pipeline.get_model_stats()
