from typing import Tuple, List
import os
import glob
from datetime import datetime

import pandas as pd
from sklearn.linear_model import SGDClassifier  # type: ignore

from disk_analyzer.stages.data_collector import DataCollector
from disk_analyzer.stages.data_stats import DataStats
from disk_analyzer.stages.model_pipeline import ModelPipeline
from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE, STORAGE_PATH, TIMES, FEATURES_NUM
from disk_analyzer.models.DLClassifier import DLClassifier


class Controller():
    """
    Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        self.model_pipeline = None
        self.start_idx = None
        self.end_idx = None
        self.mode = 'date'
        self.__is_preprocessed = False
        self.paths = []

    def collect_data(self, *args, **kwargs):
        """
        Collects data from the given paths and saves it to the given location.
        """
        DataCollector(*args, **kwargs).collect_data()

    def rebatch(self, new_batchsize: int):
        """
        Changes the batchsize of the data collected.
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
        paths = glob.glob(os.path.join(storage_path, '*.csv'))
        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline(paths)
        self.model_pipeline.preprocess(data_paths=paths, mode=model_mode)
        self.__is_preprocessed = True

    def fit(self, model_name: str = 'logistic_regression', preprocessed_path: str = PREPROCESSOR_STORAGE):
        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline()
        if model_name == 'logistic_regression':
            model = SGDClassifier(loss='log_loss')
            self.model_pipeline.set_model(model, interface='sklearn')
        elif model_name == 'NN':
            model = DLClassifier(FEATURES_NUM)  # TODO: fix constant
            self.model_pipeline.set_model(model, interface='torch')
        else:
            raise ValueError("Model name must be either 'logistic_regression' or 'NN'")
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
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])
