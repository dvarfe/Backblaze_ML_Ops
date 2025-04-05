from typing import List, Optional
import pandas as pd
from disk_analyzer.utils.constants import BATCHSIZE
from disk_analyzer.stages.data_collector import DataCollector


class Controller():
    """
    Controller class which provides methods to control the pipeline.
    """

    def __init__(self):
        pass

    def collect_data(self, *args, **kwargs):
        """
        Collects data from the given paths and saves it to the given location.
        """
        DataCollector(*args, **kwargs).collect_data()

    def rebatch(self, new_batchsize: int):
        """
        Changes the batchsize of the data collected.
        """
        DataCollector(paths=[], batchsize=new_batchsize,
                      cfgpath='').collect_data()
