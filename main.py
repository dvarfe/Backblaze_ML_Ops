import sys
import os
import shlex
import cmd
import argparse
from datetime import datetime
from disk_analyzer.controller import Controller
from disk_analyzer.view import Viewer
from disk_analyzer.utils.constants import BATCHSIZE, COLLECTOR_CFG, STORAGE_PATH
from disk_analyzer.utils.constants import STATIC_STATS, DYNAMIC_STATS, STATIC_STATS_DESCRIPTION, DYNAMIC_STATS_DESCRIPTION, MODELS_VAULT, DEFAULT_MODEL_PATH


data_collect_parser = argparse.ArgumentParser()
data_collect_parser.add_argument(
    'dirpath',
    nargs='*'
)
data_collect_parser.add_argument(
    '-b',
    '--batchsize',
    type=int,
    default=BATCHSIZE,
    dest='batchsize'
)
data_collect_parser.add_argument(
    '-s',
    '--storagepath',
    type=str,
    default=STORAGE_PATH,
    dest='storage_path'
)
data_collect_parser.add_argument(
    '-c',
    '--cfgpath',
    type=str,
    default=COLLECTOR_CFG,
    dest='cfgpath'
)


data_stats_parser = argparse.ArgumentParser()
data_stats_parser.add_argument(
    '-s',
    '--static',
    type=str,
    nargs='+',
    default=STATIC_STATS,
    dest='static_stats'
)
data_stats_parser.add_argument(
    '-d',
    '--dynamic',
    nargs='+',
    type=str,
    default=DYNAMIC_STATS,
    dest='dynamic_stats'
)
data_stats_parser.add_argument(
    '-f',
    '--figpath',
    type=str,
    default='data_stats_figures',
    dest='figpath'
)
data_stats_parser.add_argument(
    '-q',
    '--freq',
    type=str,
    default='daily',
    dest='freq'
)

save_model_parser = argparse.ArgumentParser()
save_model_parser.add_argument(
    '-p',
    '--path',
    type=str,
    default=MODELS_VAULT,
    dest='p'
)
save_model_parser.add_argument(
    '-n',
    '--name',
    type=str,
    default='default.pkl',
    dest='n'
)

load_model_parser = argparse.ArgumentParser()
load_model_parser.add_argument(
    '-p',
    '--path',
    type=str,
    default=DEFAULT_MODEL_PATH,
    dest='p'
)


class RelAnalyzer(cmd.Cmd):
    '''
    Main command loop
    '''
    prompt = '>> '

    def __init__(self, controller: Controller, viewer: Viewer):
        """App constructor.

        Args:
            controller (Controller): controller that passes al commands to other classes
            viewer (Viewer): viewer displays the results of commands
        """
        super().__init__()
        self.controller = controller
        self.viewer = viewer
        # This parameters define the exact records model works with.
        # More info in set_mode and set borders.
        self.mode = 'date'
        self.start_idx = None
        self.end_idx = None
        self.storage_path = STORAGE_PATH

    def do_EOF(self, args):
        return 1

    def do_collect_data(self, args):
        """
        Collects data from specified paths and breaks them into batches.

        If analyzer_cfg.json is located in the directory with the class, takes the parameters from there.

        Args:
            args (list): List of command-line arguments.

        Command-line arguments:
            path1, path2, path3: Paths to data sources.
            -b, --batchsize N: Set batchsize to N.
            -c, --cfgpath: Path to config file.
            -s, --storagepath: Path to storage directory

        Config file is a JSON-file of the following structure:
        {
            "batchsize": N,
            "sources": [list of sources paths]
        }
        """
        args_split = shlex.split(args)
        args_parsed = data_collect_parser.parse_args(args_split)
        self.storage_path = args_parsed.storage_path
        self.controller.collect_data(paths=list(args_parsed.dirpath),
                                     storage_path=args_parsed.storage_path,
                                     batchsize=args_parsed.batchsize,
                                     cfgpath=args_parsed.cfgpath)  # Invalid path, invalid batchsize
        print('Data succesfully collected!')

    def do_rebatch(self, args):
        """
        Change batchsize

        Command-line arguments:
        args: batchsize (int)
        """
        try:
            new_batchsize = int(args.split()[0])
        except:
            print('Incorrect value for batchsize!')
        if new_batchsize <= 0:
            print('Incorrect value for batchsize!')
        else:
            self.controller.batchsize = new_batchsize
            self.controller.rebatch(new_batchsize)
            print(
                f'Batchsize succesfully changed to {self.controller.batchsize}')

    def do_show_params(self, args):
        """Shows parameters of the data slice
        """
        print(f'Mode: {self.mode}')
        print(f'Start index: {self.start_idx}')
        print(f'End index: {self.end_idx}')

    def do_help_data_stats(self, args):
        """Shows help about each data statistics
        """
        print('Static data statistics:')
        for key in STATIC_STATS_DESCRIPTION:
            print(f'\t{key}: {STATIC_STATS_DESCRIPTION[key]}')
        print('\n')

        print('Dynamic data statistics:')
        for key in DYNAMIC_STATS_DESCRIPTION:
            print(f'\t{key}: {DYNAMIC_STATS_DESCRIPTION[key]}')

    def do_data_stats(self, args):
        """prints the statistics about the data

        Args:
            -s, --static - static statistics to calculate. 
            Usage: data_stats -s list of statistics names. 
            Whole list of statistics names can be found in stages/data_stats.py
            -d, --dynamic - dynamic statistics to calculate.
            -q, --freq - frequency of dynamic statistics(daily/monthly).
            -f, --figpath - path to directory for saving figures.
        """

        args_split = data_stats_parser.parse_args(shlex.split(args))
        figpath = args_split.figpath
        stats = self.controller.get_data_statistics(
            self.storage_path, args_split.static_stats, args_split.dynamic_stats, args_split.freq, figpath, self.mode, self.start_idx, self.end_idx)
        self.viewer.show_stats(*stats)

    def do_fit(self, args):
        """Fits model

        Args:
            args (): logistic_regression/NN/robust_regression 
        """
        args_split = shlex.split(args)
        if len(args_split) == 1:
            self.controller.fit(model_name=args_split[0])
        else:
            print('Incorrect input!')

    def do_predict(self, args):
        """Make predictions

        Accepts path to the directory with preprocessed data.
        """
        args_split = shlex.split(args)
        path = args_split[0]
        self.controller.predict(path)

    def do_predict_proba(self, args):
        """Predict probabilities
        Accepts path to the directory with preprocessed data.
        """
        args_split = shlex.split(args)
        path = args_split[0]
        self.controller.predict_proba(path)

    def do_preprocess(self, args):
        """Preprocess data
        """
        # TODO: Add arguments
        self.controller.preprocess_data()

    def do_save_model(self, args):
        """Save model
            Accepts path to the directory with models vault.
        """

        args_parsed = save_model_parser.parse_args(shlex.split(args))
        self.controller.save_model(path=args_parsed.p, name=args_parsed.n)

    def do_load_model(self, args):
        """Load model
            Accepts path to the model pickle file
        """

        args_parsed = load_model_parser.parse_args(shlex.split(args))
        self.controller.load_model(model_path=args_parsed.p)

    def do_score_model(self, args):
        args_split = shlex.split(args)
        path = args_split[0]
        self.controller.score_model(path)

    def do_exit(self, args):
        return True


if __name__ == '__main__':
    controller = Controller()
    viewer = Viewer()
    RelAnalyzer(controller, viewer).cmdloop()
