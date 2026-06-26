"""Tests for prediction aggregation utilities."""

import numpy as np
import pandas as pd
import pytest

from disk_analyzer.research.aggregation.predictions_aggregator import (
    HazardSumAgg,
    PredictionsAggregator,
)


@pytest.fixture
def sample_predictions():
    rows = []
    for sn in ["A", "B"]:
        for t in [0, 5, 10]:
            row = {"serial_number": sn, "time": t}
            for i in range(5):
                row[i] = 0.9 ** i
            rows.append(row)
    return pd.DataFrame(rows)


def test_predictions_aggregator_n_dist(sample_predictions):
    agg = PredictionsAggregator(mode="n_dist", weight=0.5)
    # Use only the latest observation per disk so timeshift is zero
    latest = sample_predictions[
        sample_predictions["time"] == sample_predictions.groupby("serial_number")["time"].transform("max")
    ]
    timeshift = latest.groupby("serial_number")["time"].transform("max") - latest["time"]
    result = agg.predict(latest, np.arange(5), timeshift=timeshift.values)
    assert "serial_number" in result.columns
    assert len(result) == latest["serial_number"].nunique()


def test_predictions_aggregator_invalid_mode():
    with pytest.raises(ValueError, match="Wrong mode"):
        PredictionsAggregator(mode="invalid")


def test_hazard_sum_agg_extended_times(sample_predictions):
    agg = HazardSumAgg()
    times = np.arange(0, 5)
    extended = agg.get_extended_times(sample_predictions, times)
    assert extended[0] == 0
    assert extended[-1] == 4
