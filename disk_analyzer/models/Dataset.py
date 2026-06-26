from typing import Tuple, List
from itertools import islice, cycle

import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

torch.manual_seed(42)
np.random.seed(42)


class DiskDataset(IterableDataset):
    def __init__(self,
                 mode: str,
                 file_paths: List[str],
                 shuffle_files: bool = True,
                 times: np.ndarray = np.arange(0, 729),
                 to_cens_time_list: List[int] = [],
                 to_term_time_list: List[int] = [],
                 cens_prob: float = -1,
                 max_buffer_size: int = 5000):
        """DiskDataset constructor.

        Args:
            mode (str): Can be train, score or infer.
            root_dir (str): Directory containing the CSV files. Defaults to PREPROCESSOR_STORAGE.
            shuffle_files (bool, optional): _description_. Defaults to True.
            to_cens_time_list (List[int]): Timeshifts to past to generate new events for.
            to_term_time_list (List[int]): Timeshifts to future to generate new events for.
            cens_prob (float): probability of censoring in data. Defaults to -1, which means no over/downsampling
        """
        self._mode = mode
        self._shuffle_files = shuffle_files
        self._file_paths = file_paths
        self.times = times
        self.to_cens_time_list = to_cens_time_list
        self.to_term_time_list = to_term_time_list

        # Buffers for censored and terminal observations
        self.cens_prob = cens_prob
        self.cens_buf = []
        self.term_buf = []

        self.feature_cols = None
        self._len = 0
        for file_path in self._file_paths:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)
            term = df['failure'].sum()
            self._len += df.shape[0] + term * (len(self.to_cens_time_list) +
                                               len(self.to_term_time_list)) - len(df[df['time'] == df['max_lifetime']])

            cur_columns = df.columns
            cur_columns = [col for col in cur_columns if col not in [
                "time", "serial_number", "max_lifetime", "failure"]]
            if self.feature_cols is None:
                self.feature_cols = cur_columns
            else:
                if self.feature_cols != cur_columns:
                    raise ValueError()  # TODO: Improve later

    def __len__(self):
        """Returns the total number of observations in the dataset."""
        return self._len

    def _iter_rows(self, file_path: str):
        if file_path.endswith(".csv"):
            for chunk in pd.read_csv(file_path, chunksize=1024):
                for row in chunk.itertuples(index=False):
                    yield [str(v) for v in row]

        else:  # parquet
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=1024):
                df = batch.to_pandas()
                for row in df.itertuples(index=False):
                    yield [str(v) for v in row]

    def __iter__(self):
        worker_info = get_worker_info()
        file_paths = self._split_files_for_workers(worker_info)

        if self._shuffle_files:
            np.random.shuffle(file_paths)

        for file_path in file_paths:
            # читаем только заголовок
            if file_path.endswith(".csv"):
                header = list(pd.read_csv(file_path, nrows=0).columns)
            else:
                header = pq.read_schema(file_path).names

            id_idx = header.index("serial_number")
            time_idx = header.index("time")

            if self._mode != "infer":
                label_idx = header.index("failure")
                event_time_idx = header.index("max_lifetime")

            for data_line in self._iter_rows(file_path):
                if self._mode == "train":
                    if data_line[event_time_idx] == data_line[time_idx]:
                        continue

                    observs = self._parse_train_line(
                        data_line, label_idx, id_idx, time_idx, event_time_idx
                    )

                    if self.cens_prob >= 0:
                        for observ in observs:
                            _, _, _, y, _ = observ
                            buf = self.term_buf if y else self.cens_buf
                            buf.append(observ)
                            if len(buf) > self.max_buffer_size:
                                buf.pop(0)

                        while self.term_buf and self.cens_buf:
                            if np.random.random() < self.cens_prob:
                                yield self.cens_buf.pop(0)
                            else:
                                yield self.term_buf.pop(0)
                    else:
                        for observ in observs:
                            yield observ

                elif self._mode == "score":
                    if data_line[event_time_idx] == data_line[time_idx]:
                        continue
                    yield self._parse_score_line(
                        data_line, label_idx, id_idx, time_idx, event_time_idx
                    )

                else:  # infer
                    yield self._parse_infer_line(data_line, id_idx, time_idx)

    def _parse_train_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> List[Tuple[str, int, torch.Tensor, bool, int]]:
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

        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [
            id_idx, time_idx, event_time_idx, label_idx]]
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time
        # data_vec += [time_to_event]
        y = (data_line[label_idx] == '1') or (data_line[label_idx] == 1) or (data_line[label_idx] == 'True')
        if y:
            extended_list = [[data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event]]
            for time in self.to_cens_time_list:
                if time_to_event - time <= 0:
                    break
                extended_list.append([data_line[id_idx], int(data_line[time_idx]),
                                     torch.tensor(data_vec), 0, time_to_event - time])
            for time in self.to_term_time_list:
                extended_list.append([data_line[id_idx], int(data_line[time_idx]),
                                     torch.tensor(data_vec), 1, time_to_event + time])
            return extended_list
        else:
            return [[data_line[id_idx], int(data_line[time_idx]), torch.tensor(data_vec), y, time_to_event]]

    def _parse_score_line(self, data_line: List[str], label_idx: int, id_idx: int, time_idx: int, event_time_idx: int) -> Tuple[str, int, torch.Tensor, bool, int]:
        """Parse a line of scoring data.

        Args:
            data_line (List[str]): A list of strings representing a line of data.
            label_idx (int): Index of the label column.
            id_idx (int): Index of the ID column.
            time_idx (int): Index of the time column.
            event_time_idx (int): Index of the event time column.

        Returns:
            Tuple[str, int, torch.Tensor, bool, int]: Parsed data including ID, time, features, label, and time to event      .
        """
        data_vec = [float(data_line[i]) for i in range(len(data_line)) if i not in [
            id_idx, time_idx, event_time_idx, label_idx]]
        y = (data_line[label_idx] == '1') or (data_line[label_idx] == 1) or (data_line[label_idx] == 'True')
        cur_time = int(data_line[time_idx])
        event_time = int(data_line[event_time_idx])
        time_to_event = event_time - cur_time

        return data_line[id_idx], cur_time, torch.Tensor(data_vec), y, time_to_event

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
        cur_time = int(data_line[time_idx])
        time_to_event = -1
        return data_line[id_idx], cur_time, torch.tensor(data_vec), 0, time_to_event

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


class DataFrameDataset(IterableDataset):
    #  Нехорошо конечно. Конвертировать датафрейм в даталоадер, только чтобы потом провернуть это обратно
    #  Но это проще и надёжнее, чем каждой модели писать новые методы. Оставим пока так.
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        times: np.ndarray = np.arange(0, 729),
        to_cens_time_list: List[int] = None,
        to_term_time_list: List[int] = None,
        cens_prob: float = -1,
        max_buffer_size: int = 5000
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self._mode = mode

        self.times = times
        self.to_cens_time_list = to_cens_time_list or []
        self.to_term_time_list = to_term_time_list or []

        self.cens_prob = cens_prob
        self.max_buffer_size = max_buffer_size

        self.cens_buf = []
        self.term_buf = []

        self.columns = list(df.columns)

        self.feature_cols = [col for col in self.columns if col not in [
            "time", "serial_number", "max_lifetime", "failure"]]

        self.id_idx = self.columns.index("serial_number")
        self.time_idx = self.columns.index("time")

        if mode != "infer":
            self.label_idx = self.columns.index("failure")
            self.event_time_idx = self.columns.index("max_lifetime")

    def __len__(self):
        """Returns the total number of observations in the dataset."""
        return len(self.df)

    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self.df)
        else:
            per_worker = int(np.ceil(len(self.df) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.df))

        df_slice = self.df.iloc[start:end]

        for row in df_slice.itertuples(index=False):

            data_line = [str(v) for v in row]

            if self._mode == "train":

                if data_line[self.event_time_idx] == data_line[self.time_idx]:
                    continue

                observs = self._parse_train_line(data_line)

                if self.cens_prob >= 0:

                    for observ in observs:
                        _, _, _, y, _ = observ
                        buf = self.term_buf if y else self.cens_buf
                        buf.append(observ)

                        if len(buf) > self.max_buffer_size:
                            buf.pop(0)

                    while self.term_buf and self.cens_buf:
                        if np.random.random() < self.cens_prob:
                            yield self.cens_buf.pop(0)
                        else:
                            yield self.term_buf.pop(0)

                else:
                    for observ in observs:
                        yield observ

            elif self._mode == "score":

                if data_line[self.event_time_idx] == data_line[self.time_idx]:
                    continue

                yield self._parse_score_line(data_line)

            else:
                yield self._parse_infer_line(data_line)

    def _parse_train_line(self, data_line):

        data_vec = [
            float(data_line[i])
            for i in range(len(data_line))
            if i not in [self.id_idx, self.time_idx,
                         self.event_time_idx, self.label_idx]
        ]

        cur_time = int(data_line[self.time_idx])
        event_time = int(data_line[self.event_time_idx])
        time_to_event = event_time - cur_time

        y = (
            data_line[self.label_idx] == '1'
            or data_line[self.label_idx] == 1
            or data_line[self.label_idx] == 'True'
        )

        if y:

            extended_list = [[
                data_line[self.id_idx],
                cur_time,
                torch.tensor(data_vec),
                y,
                time_to_event
            ]]

            for t in self.to_cens_time_list:

                if time_to_event - t <= 0:
                    break

                extended_list.append([
                    data_line[self.id_idx],
                    cur_time,
                    torch.tensor(data_vec),
                    0,
                    time_to_event - t
                ])

            for t in self.to_term_time_list:

                extended_list.append([
                    data_line[self.id_idx],
                    cur_time,
                    torch.tensor(data_vec),
                    1,
                    time_to_event + t
                ])

            return extended_list

        else:

            return [[
                data_line[self.id_idx],
                cur_time,
                torch.tensor(data_vec),
                y,
                time_to_event
            ]]

    def _parse_score_line(self, data_line):

        data_vec = [
            float(data_line[i])
            for i in range(len(data_line))
            if i not in [self.id_idx, self.time_idx,
                         self.event_time_idx, self.label_idx]
        ]

        cur_time = int(data_line[self.time_idx])
        event_time = int(data_line[self.event_time_idx])
        time_to_event = event_time - cur_time

        y = (
            data_line[self.label_idx] == '1'
            or data_line[self.label_idx] == 1
            or data_line[self.label_idx] == 'True'
        )

        return (
            data_line[self.id_idx],
            cur_time,
            torch.tensor(data_vec),
            y,
            time_to_event
        )

    def _parse_infer_line(self, data_line):

        data_vec = [
            float(data_line[i])
            for i in range(len(data_line))
            if i not in [self.id_idx, self.time_idx]
        ]

        cur_time = int(data_line[self.time_idx])

        return (
            data_line[self.id_idx],
            cur_time,
            torch.tensor(data_vec),
            0,
            -1
        )
