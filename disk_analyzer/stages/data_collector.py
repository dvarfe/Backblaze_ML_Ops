import os
import sys
import json
import pandas as pd
import glob as glob
import shutil
from typing import List, Optional
from disk_analyzer.utils.constants import BATCHSIZE


class DataCollector:
    """
    Class responsible for collecting the data.

    Accepts paths to various data sources, splits the data into batches, and copies it into storage.

    Args:
        paths (list): List of paths to various data sources.
        storage_path (str): Path to storage of batches.
        batchsize (int): Number of samples to be stored in one batch.
        cfgpath (str): Path to configuration file. If not provided, checks in the current folder.
            Configuration file has higher priority than parameters from the constructor.
            Configuration file is a JSON file with the following structure:
            {
                "batchsize": number of samples in one batch,
                "paths": list of paths to various data sources
            }
    """

    def __init__(self, paths: Optional[List[str]] = [], storage_path: str = './Data_collected', batchsize: int = BATCHSIZE, cfgpath: str = './analyzer_cfg.json'):
        if os.path.exists(cfgpath):
            with open(cfgpath) as f:
                cfg = json.load(f)
                self.batchsize = cfg['batchsize']
                self.paths = cfg['paths']
                self.storage_path = cfg['storage_path']
                self.paths += paths
        else:
            self.batchsize = batchsize
            self.paths = paths
            self.storage_path = storage_path

        if self.storage_path in self.paths:
            raise ValueError('Storage path must not be in paths')

    def __list_csv(self, paths: List[str]) -> List[str]:
        '''
        Returns a list of csv files in the paths
        '''
        csv_files = []
        for path in paths:
            files = os.listdir(path)
            csv_files += [os.path.join(path, file)
                          for file in files if file.endswith('.csv')]
        return csv_files

    def batch_resize(self):
        '''
        Rearrange the data in existing storage to match new batch size
        '''
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)
        old_files = self.__list_csv([self.storage_path])
        df_size = 0
        df_list = []
        batchnum = 0
        # files_remove = []
        for file in old_files:
            df = pd.read_csv(file)
            df_list.append(df)
            os.remove(os.path.join(
                self.storage_path, os.path.basename(file)))
            df_size += df.shape[0]
            if df_size > self.batchsize:
                df_concat = pd.concat(df_list, axis=0, ignore_index=True)
                for i in range(df_concat.shape[0] // self.batchsize):
                    new_batch = df_concat.iloc[i *
                                               self.batchsize: (i + 1) * self.batchsize, :]
                    new_batch.to_csv(os.path.join(
                        self.storage_path, f'batch_{batchnum}.csv'), index=False)
                    batchnum += 1
                if df_concat.shape[0] % self.batchsize != 0:
                    parts = df_concat.shape[0] // self.batchsize
                    df_list = [df_concat.iloc[parts * self.batchsize:, :]]
                else:
                    df_list = []
                    df_size = 0

        if len(df_list) != 0:
            df_concat = pd.concat(df_list, axis=0, ignore_index=True)
            df_concat.to_csv(os.path.join(self.storage_path,
                                          f'batch_{batchnum}.csv'), index=False)

    def collect_data(self):
        '''
        Collects the data from various sources and stores it in batchesbatches.
        Creates two categorial features: 'brand' and 'model'.
        '''
        files = self.__list_csv(self.paths)
        if not os.path.exists(self.storage_path):
            os.mkdir(self.storage_path)
        for file in files:
            df = pd.read_csv(file)
            df['date'] = df['date'].astype('datetime64[ns]')
            df['season'] = df['date'].dt.month_name()
            df.to_csv(os.path.join(self.storage_path,
                                   os.path.basename(file)), index=False)
        self.batch_resize()
