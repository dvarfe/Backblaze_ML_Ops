from typing import List, Optional
import pandas as pd
from constants import BATCHSIZE
from classes.data_collector import DataCollector


class Controller():
    def collect_data(self, *args, **kwargs):
        return DataCollector(*args, **kwargs).collect_data()
