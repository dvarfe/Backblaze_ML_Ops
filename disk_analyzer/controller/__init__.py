from typing import List, Optional
import pandas as pd
import os
from disk_analyzer.stages.data_collector import DataCollector
from disk_analyzer.stages.data_stats import DataStats


class Controller():
    """
    Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        pass

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

    def get_data_statistics(self,  storage_path, static_stats, dynamic_stats, dynamic_freq, figpath, mode, start_idx, end_idx) -> None:
        data_stats = DataStats(
            static_stats, dynamic_stats, dynamic_freq, figpath)
        df = self.open_data(storage_path, mode, start_idx, end_idx)
        stats = data_stats.calculate_stats(df)
        return stats

    def open_data(self, storage_path: str, mode: str, start_idx: int | str, end_idx: int | str) -> pd.DataFrame:
        """Open the data stored in the storage path, filter it based on the start and end index and return the dataframe.

        Args:
            storage_path (str): Path to the storage
            mode (str): Mode of operation. Can be 'batch' or 'date'.
            start_idx (int  |  str]): 
            end_idx (int  |  str]): 

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """

        contents = pd.read_csv(os.path.join(
            storage_path, 'contents.csv'))
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
            return pd.concat([pd.read_csv(path) for path in paths])
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
            df = pd.concat([pd.read_csv(path) for path in paths])
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= self.start_idx)
                    & (df['date'] <= self.end_idx)]
            return df
        else:
            raise ValueError('Mode must be either batch or date')
