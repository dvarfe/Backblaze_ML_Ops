import os
import random
from typing import Optional, Tuple, List, Generator
import glob
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np

from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE, TIMES


class DiskDataset(IterableDataset):
    def __init__(self, mode: str, file_paths: List[str], shuffle_files: bool = True, times: np.ndarray = TIMES):
        """DiskDataset constructor.

        Args:
            mode (str): Can be train, score or infer.
            root_dir (str): Directory containing the CSV files. Defaults to PREPROCESSOR_STORAGE.
            shuffle_files (bool, optional): _description_. Defaults to True.
        """
        self._mode = mode
        self._shuffle_files = shuffle_files
        self._file_paths = file_paths
        self.times = times

    def __iter__(self) -> Generator[Tuple[str, int, torch.Tensor, bool, int], None, None]:
        # Shuffle files at the start of each epoch
        worker_info = get_worker_info()
        file_paths = self._split_files_for_workers(worker_info)

        if self._shuffle_files:
            random.shuffle(file_paths)

        # Get data from files
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # Skip header
                header = f.readline().strip().split(',')
                id_idx = header.index('serial_number')
                time_idx = header.index('time')
                if self._mode != 'infer':
                    label_idx = header.index('failure')
                    event_time_idx = header.index('max_lifetime')
                for line in f:
                    data_line = line.strip().split(',')
                    if self._mode == 'train':
                        yield self._parse_train_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                    elif self._mode == 'score':
                        # We shouldn't use last observation in chain when scoring
                        if data_line[event_time_idx] == data_line[time_idx]:
                            continue
                        yield self._parse_score_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                    elif self._mode == 'infer':
                        yield self._parse_infer_line(data_line, id_idx, time_idx)

    def _parse_train_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        # Parse the line and convert it to a tensor

        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx, event_time_idx]]
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time
        data_vec += [time_to_event]
        y = bool(data_line[label_idx])
        return data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event

    def _parse_score_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx, event_time_idx]]
        y = bool(data_line[label_idx])
        lifetime = int(data_line[event_time_idx])

        return data_line[id_idx], int(data_line[time_idx]), torch.Tensor(data_vec), y, lifetime

    def _parse_infer_line(self, data_line: List[str], id_idx: int, time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx]]

        return data_line[id_idx], int(data_line[time_idx]), torch.Tensor(data_vec), 0, -1

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
