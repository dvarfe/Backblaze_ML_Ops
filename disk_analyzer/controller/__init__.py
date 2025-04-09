from typing import Tuple, List, Optional
import pandas as pd
import os
from datetime import datetime
from disk_analyzer.stages.data_collector import DataCollector
from disk_analyzer.stages.data_stats import DataStats
from disk_analyzer.stages.model_pipeline import ModelPipeline
from disk_analyzer.utils.constants import STATIC_STATS, STORAGE_PATH


class Controller():
    """
    Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        self.model_pipeline = None
        self.start_idx = None
        self.end_idx = None
        self.mode = 'date'
        self.paths = []

    def set_mode(self, mode: str):
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

    def get_data_statistics(self,  storage_path, static_stats, dynamic_stats, dynamic_freq, figpath) -> Tuple[dict, str]:
        data_stats = DataStats(
            static_stats, dynamic_stats, dynamic_freq, figpath)
        paths = self.select_data(storage_path, self.mode, self.start_idx, self.end_idx)
        df = self.open_data(paths)
        stats = data_stats.calculate_stats(df)
        return stats

    def select_data(self, storage_path: str, mode: str, start_idx: Optional[int | str], end_idx: Optional[int | str]) -> List[str]:
        """Open the data stored in the storage path, filter it based on the start and end index and return the dataframe.

        Args:
            storage_path (str): Path to the storage
            mode (str): Mode of operation. Can be 'batch' or 'date'.
            start_idx (int  |  str]): 
            end_idx (int  |  str]): 

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        contents = pd.read_csv(os.path.join(storage_path, 'contents.csv'))
        contents['min_date'], contents['max_date'] = pd.to_datetime(
            contents['min_date']), pd.to_datetime(contents['max_date'])

        self.start_idx, self.end_idx = start_idx, end_idx
        if mode == 'batch':
            if start_idx is None:
                self.start_idx = 0
            if end_idx is None:
                self.end_idx = contents['batchnum'].max()
            paths = [os.path.join(storage_path, f'{batchnum}.csv') for batchnum in contents[(
                self.start_idx <= contents['batchnum']) & (contents['batchnum'] <= self.end_idx)]['batchnum']]

        elif mode == 'date':
            if start_idx is None:
                self.start_idx = contents['min_date'].min()
            if end_idx is None:
                self.end_idx = contents['max_date'].max()

            first_batch = contents.loc[(contents['min_date'] <= self.start_idx) & (
                contents['max_date'] >= self.start_idx)]['batchnum'].iloc[0]

            last_batch = contents.loc[(contents['min_date'] <= self.end_idx) & (
                contents['max_date'] >= self.end_idx)]['batchnum'].iloc[0]
            paths = [os.path.join(storage_path, f'batch_{batchnum}.csv') for batchnum in range(
                first_batch, last_batch + 1)]
        else:
            raise ValueError('Mode must be either batch or date')
        self.paths = paths
        return self.paths

        # if mode == 'batch':
        #     df = pd.concat([pd.read_csv(path) for path in paths])
        #     df['date'] = pd.to_datetime(df['date'])
        # elif mode == 'date':
        #     df = pd.concat([pd.read_csv(path) for path in paths])
        #     df['date'] = pd.to_datetime(df['date'])
        #     df = df[(df['date'] >= self.start_idx) & (df['date'] <= self.end_idx)]
        # self.paths = paths
        # return df

    def preprocess_data(self, storage_path: str = STORAGE_PATH, model_mode: str = 'train'):
        paths = self.select_data(storage_path, self.mode, self.start_idx, self.end_idx)
        if self.model_pipeline is None:
            self.model_pipeline = ModelPipeline(paths)
        self.model_pipeline.preprocess(data_paths=paths, mode=model_mode)

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])
