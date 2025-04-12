import os
import random
from typing import Tuple, List
import glob
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info

from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE


class DiskDataset(IterableDataset):
    def __init__(self, mode: str, file_paths: List[str], shuffle_files: bool = True):
        """DiskDataset constructor.

        Args:
            mode (str): Can be either train or test.
            root_dir (str): Directory containing the CSV files. Defaults to PREPROCESSOR_STORAGE.
            shuffle_files (bool, optional): _description_. Defaults to True.
        """
        self._mode = mode
        self._shuffle_files = shuffle_files
        self._file_paths = file_paths

    def __iter__(self):
        # Shuffle files at the start of each epoch
        worker_info = get_worker_info()
        file_paths = self._split_files_for_workers(worker_info)

        if self._shuffle_files:
            random.shuffle(file_paths)

        # Get data from files
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # Skip header
                header = f.readline().split(',')
                id_idx = header.index('serial_number')
                label_idx = header.index('failure')
                for line in f:
                    line = line.strip()
                    yield self._parse_line(line, label_idx, id_idx)

    def _parse_line(self, line: str, label_idx: int, id_idx: int) -> Tuple[torch.Tensor, int]:
        # Parse the line and convert it to a tensor
        data_line = line.split(',')
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [label_idx, id_idx]]
        y = int(data_line[label_idx])
        return torch.tensor(data_vec), y

    def _split_files_for_workers(self, worker_info):
        # Split files across workers to avoid duplicates

        if worker_info is None:
            # Single-process mode
            return self._file_paths
        else:
            # Split files across workers
            return list(islice(
                cycle(self._file_paths),          # Create infinite cycle through files
                worker_info.id,                  # Unique index for each worker
                len(self._file_paths),            # Stop after all files are assigned
                worker_info.num_workers          # Step by total workers
            ))
