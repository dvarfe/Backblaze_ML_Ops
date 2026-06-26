"""Tests for observation sampling helpers."""

import pandas as pd
import pytest

from disk_analyzer.research.observation_sampling import align_gt_to_predictions, select_nth_observation


@pytest.fixture
def longitudinal_data():
    preds = pd.DataFrame({
        'serial_number': ['A', 'A', 'A', 'B', 'B'],
        'time': [10, 20, 30, 5, 15],
        's0': [0.9, 0.8, 0.7, 0.95, 0.85],
        's1': [0.8, 0.7, 0.6, 0.9, 0.8],
    })
    gt = pd.DataFrame({
        'serial_number': ['A', 'A', 'A', 'B', 'B'],
        'time': [10, 20, 30, 5, 15],
        'duration': [100, 90, 80, 200, 190],
        'failure': [False, False, True, False, True],
    })
    return preds, gt


def test_select_nth_observation_first(longitudinal_data):
    preds, _ = longitudinal_data
    result = select_nth_observation(preds, 1)
    assert len(result) == 2
    assert result.loc[result['serial_number'] == 'A', 'time'].iloc[0] == 10
    assert result.loc[result['serial_number'] == 'B', 'time'].iloc[0] == 5


def test_select_nth_observation_third(longitudinal_data):
    preds, _ = longitudinal_data
    result = select_nth_observation(preds, 3)
    assert len(result) == 2
    assert result.loc[result['serial_number'] == 'A', 'time'].iloc[0] == 30
    assert result.loc[result['serial_number'] == 'B', 'time'].iloc[0] == 15


def test_select_nth_observation_short_history(longitudinal_data):
    preds, _ = longitudinal_data
    result = select_nth_observation(preds, 10)
    assert len(result) == 2
    assert result.loc[result['serial_number'] == 'A', 'time'].iloc[0] == 30
    assert result.loc[result['serial_number'] == 'B', 'time'].iloc[0] == 15


def test_select_nth_observation_preserves_index(longitudinal_data):
    preds, gt = longitudinal_data
    result = select_nth_observation(preds, 2)
    aligned_gt = align_gt_to_predictions(result, gt)
    assert list(result.index) == list(aligned_gt.index)
    assert aligned_gt.loc[result.index[0], 'duration'] == gt.loc[result.index[0], 'duration']


def test_align_gt_to_predictions(longitudinal_data):
    preds, gt = longitudinal_data
    selected = select_nth_observation(preds, 1)
    aligned = align_gt_to_predictions(selected, gt)
    assert len(aligned) == len(selected)
    for idx in selected.index:
        assert aligned.loc[idx, 'serial_number'] == gt.loc[idx, 'serial_number']
        assert aligned.loc[idx, 'duration'] == gt.loc[idx, 'duration']
