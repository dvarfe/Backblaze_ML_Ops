import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Self
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from disk_analyzer.utils.constants import BATCHSIZE, TEST_SIZE, FEATURES_TO_REMOVE, TRAIN_SAMPLES


class TrainTestSplitter():
    """Train-test split of data. Supports initial training and further updating of training set.
    """

    def __init__(self, test_size: float = TEST_SIZE) -> None:
        """Constructor.

        Args:
            storage_paths (str, optional): paths to batched data.
            test_size (float, optional): size of test set. Defaults to TEST_SIZE.
        """

        self.__test_size = test_size
        self.__fit = False  # Flag to check whether train-test split was run at least once

    def stratified_split(self, df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test with stratification by failure.

        Args:
            df (pd.DataFrame): Data
            test_size (float, optional): Test size - value between 0 and 1. Defaults to TEST_SIZE.
            random_state (int, optional): Random state. Defaults to 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: train and test dataframes.
        """

        id_events = df.groupby('serial_number')['failure'].max().reset_index()
        train_id, test_id = train_test_split(id_events['serial_number'], test_size=test_size, random_state=42,
                                             stratify=id_events['failure'])

        df_train = df[df['serial_number'].isin(train_id)]
        df_test = df[df['serial_number'].isin(test_id)]
        return df_train, df_test

    def train_test_split(self, data_paths: List[str], rm_truncated_from_test: bool = True) -> Tuple[NDArray[np.str_], NDArray[np.str_]]:
        """Split train and test ids from dataframe and delete truncated.

        Args:
            data_paths (List[str]): List of paths to batched data
            rm_truncated_from_test (bool, optional): Whether to remove truncated disks from test sample. Defaults to True.

        Returns:
            Tuple[NDArray[str], NDArray[str]]: train and test ids.
        """
        df = self.open_data(data_paths)
        if self.__fit:
            print('Train/test split is performed as in finetuning')
            return self.update(df, rm_truncated_from_test)
        else:
            self.__fit = True
            return self.run(df, rm_truncated_from_test)

    def run(self, df: pd.DataFrame, rm_truncated_from_test: bool = True) -> Tuple[NDArray[np.str_], NDArray[np.str_]]:
        """Runs initial train-test split

        Args:
            rm_truncated_from_test (bool, optional): Whether to remove truncated disks from test sample. Defaults to True.

        Returns:
            Tuple[NDArray[str], NDArray[str]]: train and test ids
        """
        # Perform train-test split
        df_train, df_test = self.stratified_split(
            df, test_size=self.__test_size)

        self.full_train_id = df_train['serial_number'].unique()
        self.full_test_id = df_test['serial_number'].unique()

        # Remove truncated disks
        self.trunc_remover = TruncRemover()
        df_train, self.train_trunc = self.trunc_remover.fit_transform(df_train)
        self.train_id = df_train['serial_number'].unique()

        if rm_truncated_from_test:
            df_test, self.test_trunc = self.trunc_remover.fit_transform(df_test)
        self.test_id = df_test['serial_number'].unique()

        return self.train_id, self.test_id

    def find_untruncated(self, df: pd.DataFrame) -> NDArray[np.str_]:
        """Find disks that were previously truncated but are not anymore

        Args:
            df (pd.DataFrame): Data

        Returns:
            NDArray[np.str_]: list of serial numbers of untruncated disk
        """
        df_previously_truncated = df[df['serial_number'].isin(
            self.train_trunc + self.test_trunc)]
        self.trunc_remover = TruncRemover()
        df_untruncated, _ = self.trunc_remover.fit_transform(df_previously_truncated)
        return df_untruncated['serial_number'].unique()

    def update(self, df: pd.DataFrame, rm_truncated_from_test: bool = True) -> Tuple[NDArray[np.str_], NDArray[np.str_]]:
        """Update train-test split with new data.

        Args:
            df (pd.DataFrame): Data.
            rm_truncated_from_test (bool, optional): Whether to remove truncated data from test. Defaults to False.

        Returns:
            Tuple[NDArray[str], NDArray[str]]: train and test ids
        """
        # Add disks that were previously truncated but are not anymore
        untruncated = self.find_untruncated(df)
        self.train_id += list(set(untruncated).intersection(set(self.train_trunc)))
        self.test_id += list(set(untruncated).intersection(set(self.test_trunc)))

        self.train_trunc = [disk for disk in self.train_trunc if disk not in self.train_id]
        self.test_trunc = [disk for disk in self.test_trunc if disk not in self.test_id]

        # Now split disks that haven't been seen before into train and test and delete truncated
        df_new_disks = df[~df['serial_number'].isin(self.train_trunc + self.test_trunc)]
        df_new_train, df_new_test = self.stratified_split(df_new_disks)

        df_new_train, new_train_trunc = self.trunc_remover.fit_transform(df_new_train)
        self.train_id += df_new_train['serial_number'].unique()
        self.train_trunc += new_train_trunc

        if rm_truncated_from_test:
            df_new_test, new_test_trunc = self.trunc_remover.fit_transform(df_new_test)

        self.test_id += df_new_test['serial_number'].unique()
        self.test_trunc += new_test_trunc

        return self.train_id, self.test_id

    def reset(self) -> None:
        """Switches model to 'not fitted' state
        """
        self.__fit = False

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])


class DataPreprocessor():
    """Train data preprocessor class .

        Preprocessing steps:
            1. Change date column to time
            2. Delete disks with more than one event
            3. Delete duplicates(observation with the same disk number and time)
            4. Drop columns with big percentage of NaN values and explicitely specified.
            5. Impute NaN values
            6. Vectorize categorical data
            7. Standardize data
        If new data is added, previously truncated disks may be added.
    """

    def __init__(self, storage_paths: List[str], batchsize: int = BATCHSIZE, verbose: bool = True):
        """Constructor.

        Args:
            storage_paths(str, optional): paths to batched data.
            batchsize(int, optional): Number of observations in each batch. Defaults to BATCHSIZE.
            verbose(bool, optional): Whether to print logs. Defaults to True.
        """
        self.batchsize = batchsize
        self.__verbose = verbose

        self.event_times = None

        self._time_transformer = TimeTransformer()
        self._drop_doubles = DropDoubles()
        self._drop_duplicates = DropDuplicates()
        self._feature_filter = FeatureFilter(features_to_remove=FEATURES_TO_REMOVE)
        self._nan_imputer = NanImputer()
        self.categorical_encoder = CategoricalEncoder()
        self.standard_scaler = StandardScaler()
        self.random_sampler = RandomSampler()
        self.label_shifter = LabelShifter()
        self.time_labeler = TimeLabeler()

        self.__preprocessing_pipeline = {'TimeTransformer': self._time_transformer,
                                         'DropDoubles': self._drop_doubles,
                                         'DropDuplicates': self._drop_duplicates,
                                         'FeatureFilter': self._feature_filter,
                                         'Nan_imputer': self._nan_imputer,
                                         'CategoricalEncoder': self.categorical_encoder,
                                         'StandardScaler': self.standard_scaler,
                                         'RandomSampler': self.random_sampler,
                                         'LabelShifter': self.label_shifter,
                                         'TimeLabeler': self.time_labeler}

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open all csv files specified in paths

        Args:
            paths (str): Paths to files

        Returns:
            pd.DataFrame: Concatenated data
        """
        df = pd.concat([pd.read_csv(file) for file in paths], ignore_index=True)

        return df

    def fit_transform(self, X, y=None):
        for transformer in self.__preprocessing_pipeline:
            if self.__verbose:
                print(f'Fit_tr {transformer}')
            if transformer == 'StandardScaler':
                num_cols = X.columns.difference(['season', 'model', 'serial_number', 'time', 'date', 'failure', 'time'])
                X.loc[:, num_cols] = self.__preprocessing_pipeline[transformer].fit_transform(X.loc[:, num_cols])
            else:
                X = self.__preprocessing_pipeline[transformer].fit_transform(X)

        return X

    def transform(self, X, y=None) -> pd.DataFrame:
        """Preprocess data
        Args:
            X (pd.DataFrame): Data to preprocess
        Returns:
            pd.DataFrame: Preprocessed data
        """
        for transformer in self.__preprocessing_pipeline:
            if self.__verbose:
                print(f'Applying {transformer}')
            if transformer == 'StandardScaler':
                num_cols = X.columns.difference(['season', 'model', 'serial_number', 'failure', 'time'])
                X.loc[:, num_cols] = self.__preprocessing_pipeline[transformer].transform(X.loc[:, num_cols])
            else:
                X = self.__preprocessing_pipeline[transformer].transform(X)
        return X


class TimeTransformer():
    """Change date column to time in days."""

    def __init__(self, time_column: str = 'date'):
        """Constructor.

        Args:
            time_column(str, optional): Name of the date column. Defaults to 'date'.
        """

        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.time_column] = X[self.time_column].astype('datetime64[ns]')

        X.loc[:, 'time'] = (X[self.time_column] - X.groupby('serial_number')
                            [self.time_column].transform('min')).dt.days.astype(int)
        # X.loc[:, self.time_column] = pd.to_datetime(X.loc[:, self.time_column]) - self.min_date
        # X.loc[:, 'time'] = X.loc[:, self.time_column].dt.days.astype(int)

        X.drop(self.time_column, axis=1, inplace=True)
        # # X = X.rename(columns={self.time_column: 'time'})
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class DropDoubles():
    """Remove disks with more than one event.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Events for each disk
        X_events = X.groupby('serial_number')['failure'].sum()

        # Get disks with more than one event
        drop_categories = X_events[X_events > 1].index

        # Delete anomalous disks
        X = X.drop(X[X['serial_number'].isin(drop_categories)].index).reset_index(drop=True)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class DropDuplicates():
    """Delete observations happening simultaneously.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X:
                  pd.DataFrame, y=None) -> pd.DataFrame:
        X.drop_duplicates(subset=['serial_number', 'time'], inplace=True)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class TruncRemover():
    """Class to remove truncated disks.
    """

    def fit(self, X: pd.DataFrame, y=None) -> Self:
        self.last_observation = X['date'].max()
        return self

    def transform(self, X: pd.DataFrame, y=None) -> Tuple[pd.DataFrame, List[str]]:
        trunc_id = X[(X['date'] == self.last_observation) & (X['failure'] != 1)]['serial_number'].unique()
        X = X[~X['serial_number'].isin(trunc_id)]
        return X, trunc_id

    def fit_transform(self, X: pd.DataFrame, y=None) -> Tuple[pd.DataFrame, List[str]]:
        self.fit(X, y)
        return self.transform(X, y)


class FeatureFilter():
    """Classs to remove columns with lots of NaN values or explicitly specified.
    """

    def __init__(self, nan_fraction: float = 0.5, features_to_remove: Optional[List[str]] = None):
        """Constructor.

        Args:
            nan_fraction(float, optional): Acceptable rate of NaNs. Defaults to 0.5.
            features_to_remove(_type_, optional): _description_. Defaults to None.
            inplace(bool, optional): _description_. Defaults to True.
        """
        self.features_to_remove = features_to_remove
        self.nan_fraction = nan_fraction

    def fit(self, X, y=None):
        na_count = X.isna().sum()
        high_na_columns = [col_name for col_name in X.columns if na_count[col_name] > X.shape[0] * self.nan_fraction]
        self.features_to_remove = set(self.features_to_remove + high_na_columns)
        return self

    def transform(self, X, y=None):
        # Removes features with lots of NaNs, explicitly specified and those that have not been on fit call.
        self.features_to_remove = [col_name for col_name in self.features_to_remove if col_name in X.columns]
        X = X.drop(self.features_to_remove, axis=1)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class NanImputer():
    def __init__(self, fill_val: Dict | float = 0):
        self.fill_val = fill_val

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.sort_values(by=['serial_number', 'time'])

        X.loc[:, X.columns != 'serial_number'] = (X
                                                  .groupby('serial_number')
                                                  .transform('bfill')
                                                  .infer_objects(copy=False)
                                                  )
        if type(self.fill_val) is int:
            X.fillna(self.fill_val, inplace=True)
        else:
            for key, value in self.fill_val.items():
                X.loc[key] = X.loc[key].fillna(value)

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class LabelShifter():
    def fit_transform(self, X, y=None):

        X = X.sort_values(by=['serial_number', 'time'])
        X['failure'] = X.groupby('serial_number')['failure'].shift(-1)

        X = X.dropna(subset=["failure"])

        return X

    def transform(self, X, y=None):
        if 'failure' in X.columns:
            X = X.sort_values(by=['serial_number', 'time'])
            X['failure'] = X.groupby('serial_number')['failure'].shift(-1)

            X = X.dropna(subset=["failure"])

        return X


class CategoricalEncoder():
    """Class that encodes categorical columns.

        Target encoder for model and One-hot encoder for season.
    """

    def __init__(self):
        """Constructor.
        """
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.target_enc = TargetEncoder(target_type='binary',
                                        smooth='auto',
                                        cv=5,
                                        random_state=42)
        # This parameters are used if new categories appear in the test data
        self.global_mean = None
        self.season_categories = None

    def fit(self, X, y=None):

        if y is None:
            y = X['failure']

        self.model_categories = X['model'].unique()
        self.global_mean = y.mean()

        self.ohe.fit(X[['season']])
        self.target_enc.fit(X[['model']], y)

        return self

        # def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrames:

        #     # New categories in the test data are replaced with UNKNOWN
        #     X['model'] = np.where(
        #         X['model'].isin(self.model_categories),
        #         X['model'],
        #         'UNKNOWN'
        #     )

        #     model_enc = self.target_enc.transform(X[['model']])
        #     season_enc = self.ohe.transform(X[['season']])

        #     model_enc = np.filna(model_enc, self.global_mean)

        #     X.drop(['model', 'season'], axis=1, inplace=True)
        #     X = np.hstack([X, model_enc, season_enc])

        #     return X

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        # Change new categories in the test data with UNKNOWN
        X['model'] = np.where(
            X['model'].isin(self.model_categories),
            X['model'],
            'UNKNOWN'
        )

        model_enc = self.target_enc.transform(X[['model']])
        season_enc = self.ohe.transform(X[['season']])

        model_enc = np.nan_to_num(model_enc, nan=self.global_mean)
        X.drop(['model', 'season'], axis=1, inplace=True)

        numeric_data = X.values
        combined = np.hstack([numeric_data, model_enc, season_enc])

        new_columns = (
            list(X.columns) +
            ['model_encoded'] +
            list(self.ohe.get_feature_names_out(['season']))
        )

        return pd.DataFrame(combined, columns=new_columns, index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class RandomSampler():
    def __init__(self, n_samples: int = TRAIN_SAMPLES):
        self.n_samples = n_samples

    def fit_transform(self, X, y=None):
        self.shuffled_df_idx = (X[~(X['time'] == X.groupby('time')['time'].transform('max')) &
                                  ~(X['time'] == X.groupby('time')['time'].transform('min'))]
                                .sample(frac=1, random_state=42).
                                loc[:, ['serial_number']].
                                groupby('serial_number'))

        X_sampled = X.loc[self.shuffled_df_idx.head(self.n_samples - 1).index, :]

        X_sampled = pd.concat([X_sampled,
                               X[(X['time'] == X.groupby('serial_number')['time'].transform('max')) |
                                 (X['time'] == X.groupby('serial_number')['time'].transform('min'))]])
        return X_sampled

    def transform(self, X, y=None):
        return X


class TimeLabeler():
    def __init__(self):
        self.event_times = pd.Series()

    def fit_transform(self, X, y=None):
        event_times = pd.Series()
        new_event_times = X.groupby('serial_number')['time'].max()

        new_disks = list(set(new_event_times.index).difference(self.event_times.index))
        self.event_times = pd.concat([event_times, new_event_times[new_disks]])
        self.event_times = self.event_times.combine(new_event_times, max, fill_value=0)

        X['max_lifetime'] = self.event_times
        return X

    def transform(self, X, y=None):
        event_times = pd.Series()
        new_event_times = X.groupby('serial_number')['time'].max()

        new_disks = list(set(new_event_times.index).difference(self.event_times.index))
        self.event_times = pd.concat([event_times, new_event_times[new_disks]])
        self.event_times = self.event_times.combine(new_event_times, max, fill_value=0)

        X['max_lifetime'] = self.event_times
        return X
