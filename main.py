import shlex
import cmd
import argparse

from disk_analyzer.controller import Controller
from disk_analyzer.view import Viewer
from disk_analyzer.utils.constants import (BATCHSIZE, COLLECTOR_CFG, MODEL_CFG, STORAGE_PATH, STATIC_STATS,
                                           DYNAMIC_STATS, MODELS_VAULT, DEFAULT_MODEL_PATH, PREPROCESSOR_STORAGE)


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

        args_split = data_stats_parser.parse_args(shlex.split(args))
        figpath = args_split.figpath
        stats = self.controller.get_data_statistics(
            self.storage_path, args_split.static_stats, args_split.dynamic_stats, args_split.freq, figpath)
        self.viewer.show_stats(*stats)

    def do_fit(self, args):
        """Fits model

        Args:
           -m, - model to fit.
           -c, --config - path to config file
           -p, --preprocessed_data - path to directory with preprocessed data
        """
        fit_parser = argparse.ArgumentParser()
        fit_parser.add_argument(
            'm',
            type=str,
            default='NN',
            choices=['logistic_regression', 'NN'],
            nargs='?',
            help='Model to fit'
        )

        fit_parser.add_argument(
            '-c',
            '--config',
            type=str,
            default=MODEL_CFG,
            dest='c',
            help='Path to config file'
        )
        fit_parser.add_argument(
            '-p',
            '--preprocessed_data',
            type=str,
            default=PREPROCESSOR_STORAGE,
            dest='p',
            help='Path to directory with preprocessed data'
        )

        args_split = fit_parser.parse_args(shlex.split(args))
        try:
            self.controller.fit(model_name=args_split.m, cfg=args_split.c, preprocessed_path=args_split.p)
        except ValueError as v:
            print(v)

    def do_fine_tune(self, args):
        fine_tune_parser = argparse.ArgumentParser()
        fine_tune_parser.add_argument(
            '-p',
            '--preprocessed_data',
            type=str,
            default=PREPROCESSOR_STORAGE,
            dest='p',
            help='Path to directory with preprocessed data'
        )
        args_split = fine_tune_parser.parse_args(shlex.split(args))
        self.controller.fine_tune(args_split.p)

    def do_predict(self, args):
        """Make predictions

        Accepts path to the directory with preprocessed data.
        """
        args_split = shlex.split(args)
        path = args_split[0]
        self.controller.predict(path)

    def do_preprocess(self, args):
        """Preprocess data
        """
        # TODO: Add arguments
        self.controller.preprocess_data()

    def do_update_preprocessed(self, args):
        """Add new data to the existing preprocessed data
            args - path to the directory with new batched data
        """
        self.controller.update_preprocessed(args[0])

    def do_save_model(self, args):
        """Save model
            Accepts path to the directory with models vault.
        """

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

        args_parsed = save_model_parser.parse_args(shlex.split(args))
        self.controller.save_model(path=args_parsed.p, name=args_parsed.n)

    def do_load_model(self, args):
        """Load model
            Accepts path to the model pickle file
        """

        load_model_parser = argparse.ArgumentParser()
        load_model_parser.add_argument(
            '-p',
            '--path',
            type=str,
            default=DEFAULT_MODEL_PATH,
            dest='p'
        )
        args_parsed = load_model_parser.parse_args(shlex.split(args))
        self.controller.load_model(model_path=args_parsed.p)

    def do_score_model(self, args):
        args_split = shlex.split(args)
        path = args_split[0]
        ci, ibs = self.controller.score_model(path)
        viewer.show_metrics(ci, ibs)

    def do_save_best_model(self, args):
        """Save the best model based on a specified metric.

        Args:
            -m, --metric: Metric to use for selecting the best model ('ci' or 'ibs').
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-m', '--metric',
            type=str,
            choices=['ci', 'ibs'],
            required=True,
            help='Metric to use for selecting the best model.'
        )
        args_parsed = parser.parse_args(shlex.split(args))
        try:
            ci, ibs = self.controller.save_best_model(metric=args_parsed.metric, viewer=self.viewer)
            viewer.show_metrics(ci, ibs)
        except ValueError as v:
            print(f"Error: {v}")

    def do_exit(self, args):
        return True


if __name__ == '__main__':
    controller = Controller()
    viewer = Viewer()
    RelAnalyzer(controller, viewer).cmdloop()
