import os
import glob
import pandas as pd
from datetime import datetime
import random
import pickle

from ..utils.constants import MODELS_VC, DESCRIPTOR_NAME


class ModelVManager:
    """
    Model Version Manager
    """

    def __init__(self):
        if not os.path.exists(MODELS_VC):
            os.makedirs(MODELS_VC)
            versions_df = pd.DataFrame(columns=['id', 'model_name', 'timestamp', 'ci', 'ibs'])
            descriptor_path = os.path.join(MODELS_VC, DESCRIPTOR_NAME)
            versions_df.to_csv(descriptor_path, index=False)

    def save_model(self, model_pipeline):
        versions_df = pd.read_csv(os.path.join(MODELS_VC, DESCRIPTOR_NAME))
        model_id = random.randint(1, 10000)
        batches = glob.glob(os.path.join(model_pipeline.prep_storage_path, 'test', '*.csv'))
        ci, ibs = model_pipeline.score_model(batches)
        print(ci, ibs)
        versions_df.loc[len(versions_df), ['id', 'model_name', 'timestamp', 'ci', 'ibs']] = [
            model_id,
            model_pipeline.model_name,
            datetime.now(),
            ci,
            ibs
        ]
        with open(os.path.join(MODELS_VC, f'{model_id}.csv'), 'wb') as f:
            pickle.dump(model_pipeline, f)
        versions_df.to_csv(os.path.join(MODELS_VC, DESCRIPTOR_NAME), index=False)
