import os
import random
from typing import Tuple
import glob
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info

from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE


class DiskDataset(IterableDataset):
    def __init__(self, mode: str, root_dir: str = PREPROCESSOR_STORAGE, shuffle_files: bool = True):
        """DiskDataset constructor.

        Args:
            mode (str): Can be either train or test.
            root_dir (str): Directory containing the CSV files. Defaults to PREPROCESSOR_STORAGE.
            shuffle_files (bool, optional): _description_. Defaults to True.
        """
        self._mode = mode
        self._root_dir = root_dir
        self._shuffle_files = shuffle_files
        self._file_paths = glob.glob(os.path.join(root_dir, mode, '*.csv'))

    def __iter__(self):
        # Shuffle files at the start of each epoch
        worker_info = get_worker_info()
        file_paths = self._split_files_for_workers(worker_info)

        if self.shuffle_files:
            random.shuffle(file_paths)

        # Get data from files
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # Skip header
                header = f.readline()
                label_idx = header.index('time')
                for line in f:
                    line = line.strip()
                    yield self._parse_line(line, label_idx)

    def _parse_line(self, line: str, label_idx: int) -> Tuple[torch.Tensor, int]:
        # Parse the line and convert it to a tensor
        data_vec = list(map(int, line.split(',')))
        X = data_vec[:label_idx] + data_vec[label_idx + 1:]
        y = data_vec[label_idx]
        return torch.tensor(data_vec), y

    def _split_files_for_workers(self, worker_info):
        # Split files across workers to avoid duplicates

        if worker_info is None:
            # Single-process mode
            return self.file_paths
        else:
            # Split files across workers
            return list(islice(
                cycle(self.file_paths),          # Create infinite cycle through files
                worker_info.id,                  # Unique index for each worker
                len(self.file_paths),            # Stop after all files are assigned
                worker_info.num_workers          # Step by total workers
            ))
