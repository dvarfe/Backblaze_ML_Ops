"""Helpers for selecting observations when scoring longitudinal models."""

import pandas as pd


def select_nth_observation(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select the n-th chronological observation per disk (1-indexed).

    Disks with fewer than n observations keep their last available row.
    Original row indices are preserved for GT alignment.
    """
    df = df.sort_values(['serial_number', 'time'])
    return df.groupby('serial_number').head(n).groupby('serial_number').tail(1)


def align_gt_to_predictions(preds: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Align ground-truth rows to prediction rows by index."""
    return gt.loc[preds.index]
