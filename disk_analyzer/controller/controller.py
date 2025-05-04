import json
from typing import Tuple, List
import os
import glob

import pandas as pd

from ..stages import DataCollector, DataStats
from ..models import DLClassifier
from ..model_pipeline import ModelPipeline
from ..utils.constants import PREPROCESSOR_STORAGE, STORAGE_PATH, TIMES, FEATURES_NUM


class Controller:
    """Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        """
        Initialize the Controller class.

        Sets up the model pipeline and paths attributes for data processing and analysis.
        Initially sets model_pipeline to None and creates an empty list for paths.
        """
        self.model_pipeline = None
        self.paths = []

    def collect_data(self, *args, **kwargs):
        """
        Collect data using the DataCollector.

        Args:
            *args: Variable positional arguments to be passed to DataCollector.
            **kwargs: Variable keyword arguments to be passed to DataCollector.
        """
        DataCollector(*args, **kwargs).collect_data()

    def rebatch(self, new_batchsize: int):
        """
        Change the batch size of collected data.

        Args:
            new_batchsize (int): The new batch size to be used for data collection.
        """
        DataCollector(paths=[], batchsize=new_batchsize,
                      cfgpath='').collect_data()

    def get_data_statistics(self,  storage_path: str, static_stats: List[str], dynamic_stats: List[str], dynamic_freq: str, figpath: str) -> Tuple[dict, str]:
        """Returns statistics of the collected data.

        Args:
            storage_path (str): Path to the storage directory.
            static_stats (Dict[str]): Static stats to collect
            dynamic_stats (Dict[str]): Dynamic stats to collect
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

    def preprocess_data(self, storage_path: str = STORAGE_PATH, model_mode: str = 'train'):
        """
        Preprocess data for model training or inference.

        Finds all CSV files in the specified storage path and initializes 
        a ModelPipeline for data preprocessing.

        Args:
            storage_path (str, optional): Path to the directory containing data files. 
                                          Defaults to STORAGE_PATH.
            model_mode (str, optional): Mode of preprocessing, either 'train' or 'inference'. 
                                        Defaults to 'train'.
        """
        paths = glob.glob(os.path.join(storage_path, '*.csv'))
        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline(paths)
        self.model_pipeline.preprocess(data_paths=paths, mode=model_mode)
        self.__is_preprocessed = True

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

        if os.path.exists(cfg):
            model_params = json.load(open(cfg, 'r'))
        else:
            model_params = {}

        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline()
        self.model_pipeline.set_model(model_name, model_params=model_params)
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
        print(f'Predictions saved to {predictions_path}/prediction.csv')

    def score_model(self, path: str, times=TIMES):
        print(self.model_pipeline.score_model([path], times))

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Combine multiple CSV files into a single pandas DataFrame.

        Args:
            paths (List[str]): List of file paths to CSV files.

        Returns:
            pd.DataFrame: A consolidated DataFrame containing data from all input files.
        """
        return pd.concat([pd.read_csv(path) for path in paths])
