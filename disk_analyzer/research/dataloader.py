"""DataLoader helpers for research experiments."""

from typing import Optional

from torch.utils.data import DataLoader

from disk_analyzer.models.Dataset import DiskDataset
from disk_analyzer.research.config import DataConfig


def prepare_dataloader(
    train_samples: int,
    data_cfg: DataConfig,
    data_type: str,
    dataset_type: str,
    test_samples: Optional[int] = None,
    data_ext: str = "csv",
) -> DataLoader:
    if data_type == "train":
        files = [f"{data_cfg.data_folder}/{train_samples}_train_preprocessed.{data_ext}"]
    else:
        if test_samples is None:
            raise ValueError("test_samples must be provided for test loader")
        files = [f"{data_cfg.data_folder}/{train_samples}_{test_samples}_test_preprocessed.{data_ext}"]

    batch_size = data_cfg.train_batchsize if dataset_type == 'train' else data_cfg.score_batchsize

    ds = DiskDataset(
        dataset_type,
        files,
        to_cens_time_list=data_cfg.to_cens_shift,
        to_term_time_list=data_cfg.to_term_shift,
        cens_prob=data_cfg.cens_prob,
    )
    return DataLoader(ds, batch_size=batch_size)
