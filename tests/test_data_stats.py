import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from disk_analyzer.stages import DataStats


class TestDataStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment.
        """
        np.random.seed(42)

        disk_ids = np.arange(1, 101)
        start_date = datetime(2020, 1, 1)

        data = []
        for disk_id in disk_ids:
            # Generate random number of observations for each disk_id
            n_obs = np.random.randint(3, 10)

            # Generate random dates for each observation
            dates = [
                start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_obs)]
            dates.sort()

            # Generate failure flag for last observation
            failure = np.random.choice([0, 1], p=[0.7, 0.3])

            for i, date in enumerate(dates):
                event = 1 if (i == len(dates)-1 and failure) else 0
                data.append({
                    'serial_number': disk_id,
                    'date': date,
                    'failure': event,
                    'smart_1': np.random.normal(100, 10),
                    'smart_2': np.random.normal(50, 5)
                })

        cls.test_df = pd.DataFrame(data)

        # Add nan values for some observations
        cls.test_df.loc[np.random.choice(
            cls.test_df.index, 10), 'smart_1'] = np.nan

        cls.stats_analyzer = DataStats()

    def test_calculate_data_size(self):

        rows, cols = self.stats_analyzer.calculate_data_size(self.test_df)
        self.assertEqual(rows, self.test_df.shape[0])
        self.assertEqual(cols, self.test_df.shape[1])

    def test_date_calculations(self):

        min_date = self.stats_analyzer.calculate_min_date(self.test_df)
        max_date = self.stats_analyzer.calculate_max_date(self.test_df)

        self.assertEqual(min_date, self.test_df['date'].min())
        self.assertEqual(max_date, self.test_df['date'].max())

    def test_na_rate(self):

        na_rates = self.stats_analyzer.calculate_na_rate(self.test_df)
        self.assertAlmostEqual(na_rates['smart_1'],
                               self.test_df['smart_1'].isna().mean() * 100,
                               places=2)
        self.assertEqual(na_rates['serial_number'], 0)

    def test_mean_time_between_observ(self):

        mean_time = self.stats_analyzer.calculate_mean_time_between_observ(
            self.test_df)
        self.assertGreaterEqual(mean_time, 0)

    def test_calculate_stats(self):
        result, _ = self.stats_analyzer.calculate_stats(self.test_df)

        self.assertIsInstance(result, dict)
        expected_keys = ['data_size', 'min_date', 'max_date', 'mean_lifetime',
                         'max_lifetime', 'na_rate',
                         'survival_rate', 'failure_rate']
        for key in expected_keys:
            self.assertIn(key, result)

        self.assertIsInstance(result['min_date'], datetime)
        self.assertIsInstance(result['max_date'], datetime)

    def test_no_failures(self):

        no_failures_df = self.test_df.copy()
        no_failures_df['failure'] = 0
        survival_rate = self.stats_analyzer.calculate_survival_rate(
            no_failures_df)
        self.assertEqual(survival_rate, 100)


if __name__ == '__main__':
    unittest.main()
