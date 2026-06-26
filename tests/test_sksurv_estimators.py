"""Tests for scikit-survival estimators."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

from disk_analyzer.models.sksurv_estimators import (
    GradientBoostingSurvivalEstimator,
    RandomSurvivalForestEstimator,
)


def _make_dataloader(n_rows: int = 40, n_features: int = 5):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    serials = [f"s{i // 4}" for i in range(n_rows)]
    times = (np.arange(n_rows) % 4).astype(np.int32)
    y = (times == 3).astype(bool)
    durations = (4 - times).astype(np.float32)

    rows = list(zip(serials, times, X, y, durations))

    class FakeDataLoader:
        feature_cols = [f"f{i}" for i in range(n_features)]

        def __init__(self, data):
            self.data = data

        def __iter__(self):
            batch_size = 16
            for i in range(0, len(self.data), batch_size):
                chunk = self.data[i:i + batch_size]
                serials_b, times_b, X_b, y_b, dur_b = zip(*chunk)
                yield (
                    list(serials_b),
                    np.array(times_b),
                    np.stack(X_b),
                    np.array(y_b),
                    np.array(dur_b),
                )

    return FakeDataLoader(rows)


@pytest.mark.parametrize("estimator_cls", [RandomSurvivalForestEstimator, GradientBoostingSurvivalEstimator])
def test_sksurv_fit_predict(estimator_cls):
    dl_train = _make_dataloader()
    dl_score = _make_dataloader()

    model = estimator_cls(n_estimators=10, max_depth=3, random_state=42)
    model.fit(dl_train)

    times = np.arange(0, 5)
    df_pred, df_gt = model.predict(dl_score, times)

    assert not df_pred.empty
    assert not df_gt.empty
    assert list(df_pred.columns[:3]) == ["serial_number", "time", 0]
    expected = model.get_expected_time_by_predictions(df_pred, times)
    assert expected.shape[0] == len(df_pred)
