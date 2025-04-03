import sys
import os
import shlex
import cmd
import argparse
from src.controller.controller import Controller
from constants import BATCHSIZE, COLLECTOR_CFG, STORAGE_PATH

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


class RelAnalyzer(cmd.Cmd):
    '''
    Main command loop
    '''
    prompt = '>> '

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.mlpipeline = None

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

        Config file is a JSON-file of the following structure:
        {
            "batchsize": N,
            "sources": [list of sources paths]
        }
        """
        args_split = shlex.split(args)
        args_parsed = data_collect_parser.parse_args(args_split)
        self.controller.collect_data(paths=list(args_parsed.dirpath),
                                     storage_path=args_parsed.storage_path,
                                     batchsize=args_parsed.batchsize,
                                     cfgpath=args_parsed.cfgpath)
        print('Data succesfully collected!')


if __name__ == '__main__':
    controller = Controller()
    RelAnalyzer(controller).cmdloop()
