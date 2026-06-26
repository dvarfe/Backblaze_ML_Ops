"""Named aggregator presets for grid search over aggregation methods."""

from disk_analyzer.research.aggregation.predictions_aggregator import PredictionsAggregator

PRESETS = {
    "default": {
        "n_dist": {
            "0.01": PredictionsAggregator(mode='n_dist', weight=0.01),
            "0.1": PredictionsAggregator(mode='n_dist', weight=0.1),
            "0.3": PredictionsAggregator(mode='n_dist', weight=0.3),
            "0.5": PredictionsAggregator(mode='n_dist', weight=0.5),
            "0.7": PredictionsAggregator(mode='n_dist', weight=0.7),
            "0.9": PredictionsAggregator(mode='n_dist', weight=0.9),
            "0.99": PredictionsAggregator(mode='n_dist', weight=0.99),
        },
        "t_dist": {
            "0.1": PredictionsAggregator(mode='t_dist', weight=0.1),
            "1": PredictionsAggregator(mode='t_dist', weight=1),
            "10": PredictionsAggregator(mode='t_dist', weight=10),
            "25": PredictionsAggregator(mode='t_dist', weight=25),
            "50": PredictionsAggregator(mode='t_dist', weight=50),
            "100": PredictionsAggregator(mode='t_dist', weight=100),
            "1000": PredictionsAggregator(mode='t_dist', weight=1000),
        },
        "prob_dist": {
            "-1": PredictionsAggregator(mode='prob_dist'),
        },
        "geom": {
            "0.01": PredictionsAggregator(mode='geom', weight=0.01),
            "0.1": PredictionsAggregator(mode='geom', weight=0.1),
            "0.3": PredictionsAggregator(mode='geom', weight=0.3),
            "0.5": PredictionsAggregator(mode='geom', weight=0.5),
            "0.7": PredictionsAggregator(mode='geom', weight=0.7),
            "0.9": PredictionsAggregator(mode='geom', weight=0.9),
            "0.99": PredictionsAggregator(mode='geom', weight=0.99),
        },
    },
    "minimal": {
        "n_dist": {
            "0.1": PredictionsAggregator(mode="n_dist", weight=0.1),
        },
    },
}


def get_aggregators(preset: str):
    if preset not in PRESETS:
        raise ValueError(f"Unknown aggregator preset: {preset}. Available: {list(PRESETS)}")
    return PRESETS[preset]
