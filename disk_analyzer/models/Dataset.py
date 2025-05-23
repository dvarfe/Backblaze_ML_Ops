import random
from typing import Tuple, List, Generator
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np

from ..utils.constants import TIMES


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

        self._len = 0
        for file_path in self._file_paths:
            with open(file_path, 'r') as f:
                f.readline()
                self._len += sum(1 for _ in f)

    def __len__(self):
        """Returns the total number of observations in the dataset."""
        return self._len

    def __iter__(self) -> Generator[Tuple[str, int, torch.Tensor, bool, int], None, None]:
        '''
        Should always return time as the last column in data
        '''
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
                        if data_line[event_time_idx] == data_line[time_idx]:
                            continue
                        yield self._parse_train_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                    elif self._mode == 'score':
                        # We shouldn't use last observation in chain
                        if data_line[event_time_idx] == data_line[time_idx]:
                            continue
                        yield self._parse_score_line(data_line, label_idx, id_idx, time_idx, event_time_idx)
                    elif self._mode == 'infer':
                        yield self._parse_infer_line(data_line, id_idx, time_idx)

    def _parse_train_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of training data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            label_idx (int): Index of the label column.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.
            event_time_idx (int): Index of the event time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, label, and time to event.
        """
        # Parse the line and convert it to a tensor

        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx, event_time_idx]]
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time
        # data_vec += [time_to_event]
        y = data_line[label_idx] == '1'
        return data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event

    def _parse_score_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of scoring data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            label_idx (int): Index of the label column.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.
            event_time_idx (int): Index of the event time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, label, and lifetime.
        """
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx, event_time_idx]]
        y = data_line[label_idx] == '1'
        lifetime = int(data_line[event_time_idx])

        return data_line[id_idx], int(data_line[time_idx]), torch.Tensor(data_vec), y, lifetime

    def _parse_infer_line(self, data_line: List[str], id_idx: int, time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of inference data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, and placeholders for label and time to event.
        """
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [id_idx, time_idx]]

        return data_line[id_idx], int(data_line[time_idx]), torch.Tensor(data_vec), 0, -1

    def _split_files_for_workers(self, worker_info):
        """Split files across workers to avoid duplicates.

        Args:
            worker_info: Information about the current worker process.

        Returns:
            List[str]: A list of file paths assigned to the current worker.
        """
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
