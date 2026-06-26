#!/usr/bin/env python3
"""Run a grid-search training experiment from a YAML config."""

import argparse
import os
import sys

# Ensure repo root is on sys.path when invoked as scripts/run_experiment.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disk_analyzer.research.config import load_grid_search_config
from disk_analyzer.research.grid_search import run_grid_search
from disk_analyzer.research.paths import DEFAULT_EXPERIMENT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run experiment grid search")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_EXPERIMENT_CONFIG),
        help="Path to YAML config (default: configs/experiment_config.yaml)",
    )
    parser.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value")
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    cfg = load_grid_search_config(args.config)
    run_grid_search(cfg)


if __name__ == "__main__":
    main()
