import os
import glob
import pandas as pd
from datetime import datetime
import random
import pickle
import shutil

from ..utils.constants import MODELS_VC, DESCRIPTOR_NAME


class ModelVManager:
    """Model Version Manager.

    Manages saving and loading of model versions, along with associated metadata.
    """

    def __init__(self):
        """Initialize the ModelVManager class.

        Sets up the descriptor path and ensures the models version control directory exists.
        If the directory does not exist, it creates it and initializes an empty descriptor file.
        """
        self.descriptor_path = os.path.join(MODELS_VC, DESCRIPTOR_NAME)
        if not os.path.exists(MODELS_VC):
            os.makedirs(MODELS_VC)
            versions_df = pd.DataFrame(columns=['id', 'model_name', 'timestamp', 'ci', 'ibs'])
            versions_df.to_csv(self.descriptor_path, index=False)

    def save_model(self, model_pipeline):
        """Save a model along with its metadata.

        Args:
            model_pipeline: The model pipeline instance to save.
        """
        versions_df = pd.read_csv(self.descriptor_path)
        model_id = random.randint(1, 10000)
        batches = glob.glob(os.path.join(model_pipeline.prep_storage_path, 'test', '*.csv'))
        ci, ibs = model_pipeline.score_model(batches)
        model_pipeline.model_stats['CI_test'].append(ci)
        model_pipeline.model_stats['IBS_test'].append(ibs)
        print(ci, ibs)
        versions_df.loc[len(versions_df), ['id', 'model_name', 'timestamp', 'ci', 'ibs']] = [
            model_id,
            model_pipeline.model_name,
            datetime.now(),
            ci,
            ibs
        ]
        with open(os.path.join(MODELS_VC, f'{model_id}.pkl'), 'wb') as f:
            pickle.dump(model_pipeline, f)
        versions_df.to_csv(os.path.join(MODELS_VC, DESCRIPTOR_NAME), index=False)

    def save_best_model(self, metric: str, model_path: str):
        df = pd.read_csv(self.descriptor_path)
        if metric == 'ci':
            best_model_id = int(df[df['ci'] == df['ci'].max()]['id'].unique()[0])
        elif metric == 'ibs':
            best_model_id = int(df[df['ibs'] == df['ibs'].max()]['id'].unique()[0])
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        shutil.copy(os.path.join(MODELS_VC, f'{best_model_id}.pkl'), model_path)
