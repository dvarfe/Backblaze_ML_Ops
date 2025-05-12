from math import ceil
import os
from typing import Dict, List

from matplotlib import pyplot as plt


class Viewer():
    def show_stats(self, static_stats: Dict[str, float], dynamic_figpath: str):
        """Show statistics of the data. Save plots of dynamic statistics by dynamic_figpath.

        Args:
            static_stats (List[str]): Dictionary of calculated static statistics.
            dynamic_figpath (str): Path to the directory to save figures.
        """
        print('STAIC STATISTICS:')
        for key in static_stats:
            if key == 'na_rate':
                print(f'na rate for each column:')
                print(static_stats[key])
            else:
                if type(static_stats[key]) is float:
                    print(f'{key}: {static_stats[key]:.2f}')
                else:
                    print(f'{key}: {static_stats[key]}')
        print(f'DYNAMIC STATISTICS are stored at: {dynamic_figpath}')

    def show_metrics(self, ci: float, ibs: float):
        """Show metrics

        Args:
            ci (float): Concordance Index
            ibs (float): Integrated Brier Score
        """
        print(f'Concordance Index:{ci:4f}')
        print(f'IBS: {ibs:4f}')

    def make_report(self, stats: Dict[str, List[float]], path: str) -> str:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        rows = int(len(stats) ** 0.5)
        cols = ceil(len(stats) / rows)
        plt.figure(figsize=(20, 20))
        fig, ax = plt.subplots(rows, cols)
        fig.tight_layout()
        fig.suptitle('Metrics')

        for idx, stat in enumerate(stats):
            cur_ax = ax[idx // cols, idx % cols]
            cur_ax.plot(stats[stat])
            cur_ax.set_title(stat)

        plt.savefig(path)
        return path
