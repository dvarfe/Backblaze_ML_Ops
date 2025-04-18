
import pandas as pd
from lifelines.utils import concordance_index  # type: ignore
from survivors.metrics import ibs_remain


class ModelScorer():

    def get_ci_and_ibs(self, model, df_pred, df_gt, times):

        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times
        )
        return ci, ibs
