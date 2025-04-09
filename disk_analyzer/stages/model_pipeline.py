import shutil  # Не забудьте импортировать в начале файла
import os
import shutil
import pandas as pd

from typing import Optional, Dict, List

from disk_analyzer.utils.constants import MODEL_TYPES, PREPROCESSOR_STORAGE, BATCHSIZE
from disk_analyzer.stages.data_preprocessor import TrainTestSplitter, DataPreprocessor


class ModelPipeline:
    """Class that incorporates all pipeline logic
        Consists of the following stages:
            1. Data Preprocessor: split data into train and test, preprocess data
            2. Model: train and predict
            3. Scoring: get metrics
    """

    def __init__(self,
                 data_paths: List[str],
                 model: Optional[MODEL_TYPES] = None,
                 train_test_splitter: Optional[TrainTestSplitter] = None,
                 data_preprocessor: Optional[DataPreprocessor] = None,
                 rm_truncated_from_test: bool = True,
                 batchsize: int = BATCHSIZE,
                 prep_storage_path: str = PREPROCESSOR_STORAGE,
                 model_params: Optional[Dict] = None):

        self.__model = model
        self.batchsize = batchsize
        self.prep_storage_path = prep_storage_path
        self.data_paths = data_paths
        self.__train_test_splitter = train_test_splitter or TrainTestSplitter()
        self.__data_preprocessor = data_preprocessor or DataPreprocessor(storage_paths=data_paths)
        self.__rm_truncated_from_test = rm_truncated_from_test

    def open_data(self, paths: List[str]) -> pd.DataFrame:
        """Open data from the given paths.

        Args:
            paths (List[str]): Paths.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        return pd.concat([pd.read_csv(path) for path in paths])

    def preprocess(self, data_paths: List[str], mode: str = 'train') -> None:
        """Preprocess data

        Args:
            data_paths (pd.DataFrame): Paths to data files.
            mode (str, optional): Whether to preprocess train/test/tune data. Defaults to 'train'.

        """
        if mode == 'train':
            # Split data into train and test
            train_id, test_id = self.__train_test_splitter.train_test_split(data_paths)

            # Open data and pass it to preprocessor
            df = self.open_data(data_paths)
            df_train = df[df['serial_number'].isin(train_id)]
            df_test = df[~df['serial_number'].isin(test_id)]

            df_train, df_test = self.__data_preprocessor.fit_transform(
                df_train), self.__data_preprocessor.transform(df_test)

            # print('Save Preprocessed Data')
            # for sample_name, df in zip(['Train', 'Test'], [df_train, df_test]):
            #     if not os.path.exists(os.path.join(self.prep_storage_path, sample_name)):
            #         os.
            #     df.sort_values(by=['serial_number', 'time'], inplace=True)
            #     for i in range((df.shape[0] + self.batchsize - 1)//self.batchsize):
            #         df.iloc[i*self.batchsize:(i+1)*self.batchsize].to_csv(os.path.join(
            #             self.prep_storage_path, sample_name, f'{str(i)}_preprocessed.csv'), index=False)

            print('Save Preprocessed Data')

            for sample_name, df in zip(['Train', 'Test'], [df_train, df_test]):
                sample_dir = os.path.join(self.prep_storage_path, sample_name)

                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)

                os.makedirs(sample_dir, exist_ok=True)

                df.sort_values(by=['serial_number', 'time'], inplace=True)

                total_rows = df.shape[0]
                for i in range((total_rows + self.batchsize - 1) // self.batchsize):
                    batch = df.iloc[i*self.batchsize: (i+1)*self.batchsize]
                    batch.to_csv(os.path.join(sample_dir, f'{i}_preprocessed.csv'), index=False)
        elif mode == 'test' or mode == 'tune':
            _ = self.__data_preprocessor.fit_transform(self.open_data(data_paths))

    def fit(self):
        """Fit model
        """
        pass

    def predict(self):
        """Return predictions
        """
        pass

    def predict_proba(self):
        """Return predictions
        """
        pass
