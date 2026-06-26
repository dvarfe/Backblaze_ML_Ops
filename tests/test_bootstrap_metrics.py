"""Tests for ModelScorer.bootstrap_metrics."""

import numpy as np
import pandas as pd
import pytest

from disk_analyzer.stages.model_scoring import ModelScorer


class _StubModel:
    def get_expected_time_by_predictions(self, df_pred, times):
        return df_pred['s0'].values * 100


@pytest.fixture
def sample_pred_gt():
    times = np.array([0, 1, 2])
    n = 20
    preds = pd.DataFrame({
        'serial_number': [f'D{i}' for i in range(n)],
        'time': [0] * n,
        's0': np.linspace(0.9, 0.5, n),
        's1': np.linspace(0.8, 0.4, n),
        's2': np.linspace(0.7, 0.3, n),
    })
    gt = pd.DataFrame({
        'serial_number': [f'D{i}' for i in range(n)],
        'time': [0] * n,
        'duration': np.linspace(100, 10, n),
        'failure': [i % 2 == 0 for i in range(n)],
    })
    return preds, gt, times


def test_bootstrap_metrics_length(sample_pred_gt):
    preds, gt, times = sample_pred_gt
    scorer = ModelScorer()
    results = scorer.bootstrap_metrics(
        _StubModel(), preds, gt, times, {'ci', 'ibs'}, n_bootstrap=5, seed=42,
    )
    assert len(results) == 5


def test_bootstrap_metrics_keys(sample_pred_gt):
    preds, gt, times = sample_pred_gt
    scorer = ModelScorer()
    results = scorer.bootstrap_metrics(
        _StubModel(), preds, gt, times, {'ci', 'ibs'}, n_bootstrap=3, seed=42,
    )
    for row in results:
        assert 'ci' in row
        assert 'ibs' in row


def test_bootstrap_metrics_reproducible(sample_pred_gt):
    preds, gt, times = sample_pred_gt
    scorer = ModelScorer()
    kwargs = dict(
        model=_StubModel(), df_pred=preds, df_gt=gt, times=times,
        metrics={'ci', 'ibs'}, n_bootstrap=4, seed=123,
    )
    first = scorer.bootstrap_metrics(**kwargs)
    second = scorer.bootstrap_metrics(**kwargs)
    assert first == second


def test_bootstrap_metrics_invalid_n(sample_pred_gt):
    preds, gt, times = sample_pred_gt
    scorer = ModelScorer()
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        scorer.bootstrap_metrics(
            _StubModel(), preds, gt, times, {'ci'}, n_bootstrap=0, seed=42,
        )


def test_bootstrap_metrics_mismatched_lengths(sample_pred_gt):
    preds, gt, times = sample_pred_gt
    scorer = ModelScorer()
    with pytest.raises(ValueError, match="same number of rows"):
        scorer.bootstrap_metrics(
            _StubModel(), preds, gt.iloc[:2], times, {'ci'}, n_bootstrap=2, seed=42,
        )
