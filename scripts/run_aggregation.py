#!/usr/bin/env python3
"""Run aggregation evaluation from a YAML config."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disk_analyzer.research.aggregation.quantile_runner import run_quantile_aggregation
from disk_analyzer.research.aggregation.runner import run_aggregation
from disk_analyzer.research.config import load_agg_config, load_quantile_agg_config
from disk_analyzer.research.paths import DEFAULT_AGGREGATION_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run aggregation experiment")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_AGGREGATION_CONFIG),
        help="Path to YAML config (default: configs/aggregation_config.yaml)",
    )
    parser.add_argument("--mode", choices=["standard", "quantile"], default="standard")
    parser.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if args.mode == "quantile":
        cfg = load_quantile_agg_config(args.config)
        run_quantile_aggregation(cfg)
    else:
        cfg = load_agg_config(args.config)
        run_aggregation(cfg)


if __name__ == "__main__":
    main()
