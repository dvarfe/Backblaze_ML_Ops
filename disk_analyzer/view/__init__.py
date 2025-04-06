class Viewer():
    def show_stats(self, static_stats, dynamic_figpath):
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
