from typing import Tuple

import pandas as pd
import numpy as np
from lifelines.utils import concordance_index  # type: ignore
from survivors.metrics import ibs_remain


class ModelScorer():

    def get_ci_and_ibs(self, model, df_pred:pd.DataFrame, df_gt:pd.DataFrame, times:np.ndarray) -> Tuple[float, float]:
        """Calculate Concordance Index (CI) and Integrated Brier Score (IBS).

        Args:
            model: The trained model used for predictions.
            df_pred (pd.DataFrame): DataFrame containing predicted survival functions.
            df_gt (pd.DataFrame): DataFrame containing ground truth durations and event indicators.
            times (np.ndarray): Array of time points for evaluation.

        Returns:
            Tuple[float, float]: Concordance Index (CI) and Integrated Brier Score (IBS).
        """
        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times
        )
        return ci, ibs
