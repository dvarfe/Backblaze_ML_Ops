from typing import Dict


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

    def show_metrics(self, ci, ibs):
        print(f'Concordance Index:{ci:4.f}')
        print(f'IBS: {ibs:4.f}')
