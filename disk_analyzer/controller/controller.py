from typing import List, Optional
import pandas as pd
from disk_analyzer.utils.constants import BATCHSIZE
from stages.data_collector import DataCollector


class Controller():
    def collect_data(self, *args, **kwargs):
        return DataCollector(*args, **kwargs).collect_data()
