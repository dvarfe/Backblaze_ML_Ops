from typing import Tuple, List
import os
import glob
from datetime import datetime

import pandas as pd
from sklearn.linear_model import SGDClassifier  # type: ignore

from disk_analyzer.stages.data_collector import DataCollector
from disk_analyzer.stages.data_stats import DataStats
from disk_analyzer.stages.model_pipeline import ModelPipeline
from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE, STORAGE_PATH
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

    def set_mode(self, mode: str):
        """Set the mode of the pipeline. Defines the logic by which data to process is determined.

        Args:
            mode (str): Can be either date or batch. Default is date.
        """
        if mode == 'date' or mode == 'batch':
            if self.mode != mode:
                self.start_idx = None
                self.end_idx = None
            self.mode = mode
            print(f'Mode succesfully changed to {self.mode}')
        else:
            print('Incorrect value for mode!')

    def set_borders(self, start_idx: int | str, end_idx: int | str):
        if self.mode == 'date':
            try:
                if start_idx == '-1':
                    self.start_idx = None
                else:
                    self.start_idx = datetime.strptime(str(start_idx), '%Y-%m-%d')
                if end_idx == '-1':
                    self.end_idx = None
                else:
                    self.end_idx = datetime.strptime(str(end_idx), '%Y-%m-%d')
            except:
                print('Incorrect value for borders!')
        elif self.mode == 'batch':
            try:
                if start_idx == '-1':
                    self.start_idx = None
                else:
                    self.start_idx = int(start_idx)
                if end_idx == '-1':
                    self.end_idx = None
                else:
                    self.end_idx = int(end_idx)
            except:
                print('Incorrect value for borders!')

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
            model = DLClassifier(12)  # TODO: fix constant
            self.model_pipeline.set_model(model, interface='torch')
        elif model_name == 'robust_regression':
            model = SGDClassifier(loss='modified_huber')
            self.model_pipeline.set_model(model)
        else:
            raise ValueError("Model name must be either 'logistic_regression', 'NN' or 'robust_regression'.")
        batches = glob.glob(os.path.join(preprocessed_path, 'train', '*.csv'))
        self.model_pipeline.fit(batches)

    def predict(self, path: str):
        """Model inference on a single file. The results are saved in the 'Predictions' folder.

        Args:
            path (str): Input file
        """
        predictions_path = 'Predictions/'
        df_pred = self.model_pipeline.predict([path])
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        df_pred.to_csv(f'{predictions_path}/prediction.csv', index=False)
        print(f'Predictions saved to {predictions_path}/prediction.csv')

    def predict_proba(self, path: str):
        """Model inference on a single file. The results are saved in the 'Predictions' folder.
            Returns probabilities of event for each event in the input file.
        Args:
            path (str): Input file
        """
        predictions_path = 'Predictions/'
        pred = self.model_pipeline.predict_proba([path])
        if not os.path.exists(predictions_path):
            os.mkdir(predictions_path)
        pred.to_csv(f'{predictions_path}/prediction.csv', index=False)
        print(f'Predictions saved to {predictions_path}/prediction.csv')

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])
