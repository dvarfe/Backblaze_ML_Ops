from typing import List, Tuple
from datetime import datetime
import pandas as pd
from numpy.typing import NDArray

from ..utils.constants import STATIC_STATS, DYNAMIC_STATS


class DataStats:
    def __init__(self, static_stats: List[str] = STATIC_STATS, dynamic_stats: List[str] = DYNAMIC_STATS,
                 dynamic_stats_freq: str = 'daily', fig_path: str = 'data_stats_figures') -> None:
        """Class to calculate data statistics
        Statistics can be static and dynamic.
        Static statistics are:
            data_size - number of observations in the data.
            min_date - date of the first observation.
            max_date - date of the last observation.
            mean_lifetime - mean lifetime of disks.
            max_lifetime - maximum lifetime of disks.
            na_rate - rate of missing data in each column.
            truncated_rate - rate of truncated disks.
            survival_rate - rate of survived disks.
            failure_rate - rate of failed disks.
            double_failures - rate of disks that have failed more than once.
            mean_time_between_observ - mean time between observations of a disk.
            mean_observ_per_day - mean number of observations per day.
        Dynamic statistics are calucated per day/month. They are represented as figures saved by specified path:
            mean_lifetime - mean lifetime of disks of disks alive at the day/month.
            max_lifetime - maximum lifetime of disks alive at the day/month.
            na_rate - rate of nans.
            survival_rate - rate of disks that have survived at the day/month.
            failure_rate - rate of disks that have failed at the day/month.
            mean_observ_per_day - mean number of observations.
        Args:
            static_stats (List[str], optional): List of static statistics to calculate. Defaults to STATIC_STATS.
            dynamic_stats (List[str], optional): List of dynamic statistics to calculate. Defaults to DYNAMIC_STATS.
            dynamic_stats_freq (str): Can be 'daily' or 'monthly'. Determines dynamic_stats aggregation frequency.
                                    Defaults to 'daily'.
            fig_path (str) : Path to the folder where to save the figures. Defaults to 'data_stats_figures'.
        """
        self.__static_stats = static_stats
        self.__dynamic_stats = dynamic_stats
        self.__dynamic_stats_freq = dynamic_stats_freq
        self.fig_path = fig_path

        self.__static_stats_funcs = {
            'data_size': self.calculate_data_size,
            'min_date': self.calculate_min_date,
            'max_date': self.calculate_max_date,
            'mean_lifetime': self.calculate_mean_lifetime,
            'max_lifetime': self.calculate_max_lifetime,
            'na_rate': self.calculate_na_rate,
            'truncated_rate': self.__calculate_truncated_rate,
            'survival_rate': self.calculate_survival_rate,
            'failure_rate': self.calculate_failure_rate,
            'double_failures': self.calculate_double_failures,
            'mean_time_between_observ': self.calculate_mean_time_between_observ,
            'mean_observ_per_day': self.calculate_mean_observ_per_day
        }

        # self.__dynamic_stats_funcs = {
        #     'mean_lifetime': self.calculate_dynamic_mean_lifetime,
        #     'max_lifetime': self.calculate_dynamic_max_lifetime,
        #     'na_rate': self.calculate_dynamic_na_rate,
        #     'survival_rate': self.calculate_dynamic_survival_rate,
        #     'failure_rate': self.calculate_dynamic_failure_rate,
        #     'mean_observ_per_day': self.calculate_dynamic_mean_observ_per_day
        # }

    def __delete_truncated(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NDArray]:
        """Method to delete truncated disks from the dataframe.

        Args:
            df (pd.DataFrame): Dataframe to delete from.
        """
        last_observation = df['date'].max()
        trunc_id = df[(df['date'] == last_observation) &
                      (df['failure'] != 1)]['serial_number'].unique()
        return df[~df['serial_number'].isin(trunc_id)], trunc_id

    def calculate_stats(self, df: pd.DataFrame) -> Tuple[dict, str]:
        """Method to calculate statistics.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            dict: Static statistics
            str: Path to the folder with figures of dynamic statistics
        """
        self.__data = df
        self.__data['date'] = pd.to_datetime(self.__data['date'])
        self.__data, self.__truncated_disks = self.__delete_truncated(
            self.__data)
        static_stats = self.calculate_static_stats(df)
        # self.calculate_dynamic_stats()
        return static_stats, self.fig_path

    def calculate_static_stats(self, df: pd.DataFrame) -> dict:
        """Method to calculate static statistics.

        Args:
            df (pd.DataFrame): Data.
        """
        static_stats = {}
        for stat_name in self.__static_stats:
            static_stats[stat_name] = self.__static_stats_funcs[stat_name](df)
        return static_stats

    def calculate_data_size(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Calculates the dimensions of the dataset.

        Args:
            df: Input dataframe

        Returns:
            Tuple containing (number_of_rows, number_of_columns)
        """
        return df.shape[0], df.shape[1]

    def calculate_min_date(self, df: pd.DataFrame,) -> datetime:
        """Finds the earliest date in the dataset.

        Args:
            df: Input dataframe

        Returns:
            Minimum datetime value found
        """
        return df['date'].min()

    def calculate_max_date(self, df: pd.DataFrame) -> datetime:
        """Finds the latest date in the dataset.

        Args:
            df: Input dataframe

        Returns:
            Maximum datetime value found
        """
        return df['date'].max()

    def calculate_durations(self, df: pd.DataFrame) -> pd.Series:
        """Calculates lifetimes of each disk that have failed.

        Args:
            df (pd.DataFrame): Data.

        Returns:
            pd.Series: Lifitimes.
        """
        disks_died = df[df['failure'] == 1]['serial_number'].unique()
        durations = (df[df['serial_number'].isin(disks_died)].groupby('serial_number')['date'].max(
        ) - df[df['serial_number'].isin(disks_died)].groupby('serial_number')['date'].min()).dt.days
        return durations

    def calculate_mean_lifetime(self, df: pd.DataFrame) -> float:
        """Calculates average lifetime.

        Args:
            df: Input datafram

        Returns:
            Mean lifetime value
        """
        durations = self.calculate_durations(df)
        return durations.mean()

    def calculate_max_lifetime(self, df: pd.DataFrame) -> float:
        """Finds maximum lifetime value.

        Args:
            df: Input dataframe

        Returns:
            Maximum lifetime value
        """
        durations = self.calculate_durations(df)
        return durations.max()

    def calculate_na_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculates percentage of missing values.

        Args:
            df: Input dataframe

        Returns:
            Percentage of NA values (0-100)
        """
        return df.isna().mean() * 100

    def __calculate_truncated_rate(self, df: pd.DataFrame) -> float:
        """Calculates percentage of truncated disks.

        Args:
            df: Input dataframe

        Returns:
            Percentage of truncated observations (0-100)
        """
        return self.__truncated_disks.shape[0] / df['serial_number'].nunique() * 100

    def calculate_survival_rate(self, df: pd.DataFrame) -> float:
        """Calculates survival rate. Number of disks failed divided by total number of disks.

        Args:
            df: Input dataframe

        Returns:
            Survival rate (0-100)
        """
        df_grouped = df.groupby('serial_number')['failure'].max() == 0
        disks_alive = len(df_grouped.index[df_grouped])
        return disks_alive / df['serial_number'].nunique() * 100

    def calculate_failure_rate(self, df: pd.DataFrame) -> float:
        """Calculates failure rate.

        Args:
            df: Input dataframe
            event_col: Name of event indicator column

        Returns:
            Failure rate (0-100)
        """
        return 100 - self.calculate_survival_rate(df)

    def calculate_double_failures(self, df: pd.DataFrame) -> int:
        """Counts double failure events per subject.

        Args:
            df: Input dataframe

        Returns:
            Number of disks with multiple failure events
        """
        return df.groupby('serial_number')['failure'].sum().gt(1).sum()

    def calculate_mean_time_between_observ(self, df: pd.DataFrame) -> float:
        """Calculates average time between observations.

        Args:
            df: Input dataframe

        Returns:
            Mean time between observations in days
        """
        return (df.groupby('serial_number')['date'].diff().dt.days.mean())

    def calculate_mean_observ_per_day(self, df: pd.DataFrame) -> float:
        """Calculates average observations per day.

        Args:
            df: Input dataframe

        Returns:
            Mean number of observations per day
        """
        return df.groupby(df['date'].dt.date).size().mean()
