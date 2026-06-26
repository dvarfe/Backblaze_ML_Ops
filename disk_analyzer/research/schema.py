"""CSV schema helpers for experiment result tables."""

from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import ParameterGrid


def make_schema(
    metrics: List[str],
    hparams: List[str],
    score_sample_grid: Optional[List[int]] = None,
    bootstrap_n: Optional[int] = None,
) -> Dict[str, list]:
    schema = {
        'train_samples': None,
        'method': [],
        'model_id': None,
        'train_time': None,
        'test_time': None,
        'error': None,
        'error_text': None,
    }
    if score_sample_grid is not None or bootstrap_n is not None:
        schema['test_samples'] = None
    if score_sample_grid is not None:
        schema['score_samples'] = None
    if bootstrap_n is not None:
        schema['bootstrap_iter'] = None
    for m in metrics:
        schema[f'{m}_train_same_size'] = None
        schema[f'{m}_train_max_size'] = None
        schema[f'{m}_test'] = None
    for hp in hparams:
        schema[hp] = None
    return schema


def write_dict(filename: str, d: Dict[str, list]) -> None:
    pd.DataFrame(d).to_csv(filename, mode='a', header=False, index=False)


def create_res_file(filename: str, schema: Dict[str, list]) -> None:
    pd.DataFrame(schema).to_csv(filename, index=False)


def collect_all_hparams(param_grids: Dict[str, ParameterGrid]) -> List[str]:
    hparams = set()
    for grid in param_grids.values():
        for params in grid:
            hparams.update(params.keys())
    return sorted(hparams)
