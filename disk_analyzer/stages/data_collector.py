import os
import json
import glob as glob
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from ..utils.constants import BATCHSIZE


class DataCollector:
    """Class responsible for collecting the data.

    Accepts paths to various data sources, splits the data into batches, and copies it into storage.
    """

    def __init__(self, paths: Optional[List[str]] = [], storage_path: str = './Data_collected', batchsize: int = BATCHSIZE, cfgpath: str = './analyzer_cfg.json'):
        """Initialize the DataCollector class.

        Args:
            paths (Optional[List[str]]): List of paths to various data sources.
            storage_path (str): Path to storage of batches.
            batchsize (int): Number of samples to be stored in one batch.
            cfgpath (str): Path to configuration file. If not provided, checks in the current folder.
                Configuration file has higher priority than parameters from the constructor.
        """
        if os.path.exists(cfgpath):
            with open(cfgpath) as f:
                cfg = json.load(f)
                self._batchsize = cfg['batchsize']
                self._paths = cfg['paths']
                self._storage_path = cfg['storage_path']
                self._paths += paths
        else:
            self._batchsize = batchsize
            self._paths = paths
            self._storage_path = storage_path

        if self._storage_path in self._paths:
            raise ValueError('Storage path must not be in paths')

    def _list_csv(self, paths: List[str]) -> List[str]:
        """Returns a list of csv files in the paths.

        Args:
            paths (List[str]): Paths to search for csv files.

        Returns:
            List[str]: CSV files found in the paths.
        """
        csv_files = []
        for path in paths:
            files = os.listdir(path)
            csv_files += [os.path.join(path, file)
                          for file in files if file.endswith('.csv')]
        return csv_files

    def batch_resize(self):
        """Rearrange the data in existing storage to match new batch size.
        """
        if not os.path.exists(self._storage_path):
            os.mkdir(self._storage_path)
        print('Batching!')
        # additional file to save information about batches
        df_contents = pd.DataFrame(
            columns=['batchnum', 'min_date', 'max_date'])

        old_files = self._list_csv([self._storage_path])
        old_files.sort()

        df_size = 0
        df_list = []
        batchnum = 0

        for idx, file in tqdm(enumerate(old_files)):

            df = pd.read_csv(file)
            df['date'] = df['date'].astype('datetime64[ns]')
            df_list.append(df)
            os.remove(os.path.join(
                self._storage_path, os.path.basename(file)))
            df_size += df.shape[0]

            if df_size > self._batchsize:
                df_concat = pd.concat(df_list, axis=0, ignore_index=True)
                df_concat.sort_values(by='date', inplace=True)

                for i in range(df_concat.shape[0] // self._batchsize):
                    new_batch = df_concat.iloc[i *
                                               self._batchsize: (i + 1) * self._batchsize, :]
                    new_batch.to_csv(os.path.join(
                        self._storage_path, f'batch_{batchnum}.csv'), index=False)
                    batchnum += 1

                if df_concat.shape[0] % self._batchsize != 0:
                    parts = df_concat.shape[0] // self._batchsize
                    df_list = [df_concat.iloc[parts * self._batchsize:, :]]
                else:
                    df_list = []
                    df_size = 0

        if len(df_list) != 0:
            df_concat = pd.concat(df_list, axis=0, ignore_index=True)
            df_concat.to_csv(os.path.join(self._storage_path,
                                          f'batch_{batchnum}.csv'), index=False)

    def collect_data(self):
        """Collects the data from various sources and stores it in batches.
        Creates categorial feature 'season'.
        """
        print('Begin preparation')
        files = self._list_csv(self._paths)
        if not os.path.exists(self._storage_path):
            os.mkdir(self._storage_path)
        # TODO: process several files at once
        for file in files:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            df['season'] = df['date'].dt.month_name()
            df.to_csv(os.path.join(self._storage_path,
                                   os.path.basename(file)), index=False)
        print('End preparation')

        self.batch_resize()
