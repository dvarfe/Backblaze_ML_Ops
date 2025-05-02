from typing import List
from datetime import datetime
import pandas as pd
import json
import shutil
import os
import sys
import unittest

from ..stages import DataCollector


class TestDataCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Create test data
        cls.test_dir = "test_data"
        cls.storage_path = os.path.join(cls.test_dir, "storage")
        cls.config_path = os.path.join(cls.test_dir, "test_config.json")
        cls.sample_data = [
            {"date": "2013-04-10", "model": "Hitachi HDS5C3030ALA630",
                "smart_1": 3000592982016},
            {"date": "2013-04-10", "model": "Hitachi HDS722020ALA330",
                "smart_1": 2000398934016},
        ]

        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, "source1"), exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, "source2"), exist_ok=True)

        # Create csv files in diferent sources
        pd.DataFrame(cls.sample_data).to_csv(
            os.path.join(cls.test_dir, "source1", "data1.csv"), index=False
        )
        pd.DataFrame(cls.sample_data).to_csv(
            os.path.join(cls.test_dir, "source2", "data2.csv"), index=False
        )

        # Create json config file
        config = {
            "batchsize": 3,
            "paths": [os.path.join(cls.test_dir, "source1")],
            "storage_path": cls.storage_path,
        }
        with open(cls.config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        if os.path.exists(self.storage_path):
            shutil.rmtree(self.storage_path)

    def test_initialization_with_config(self):
        collector = DataCollector(
            paths=["mock_path"],
            storage_path="mock_storage_path",
            batchsize=20,
            cfgpath=self.config_path,
        )

        self.assertEqual(collector.batchsize, 3)
        self.assertIn(os.path.join(self.test_dir, "source1"), collector.paths)
        self.assertEqual(collector.storage_path, self.storage_path)

    def test_initialization_without_config(self):
        collector = DataCollector(
            paths=["source1", "source2"],
            storage_path=self.storage_path,
            batchsize=2,
        )
        self.assertEqual(collector.batchsize, 2)
        self.assertListEqual(collector.paths, ["source1", "source2"])
        self.assertEqual(collector.storage_path, self.storage_path)

    def test_collect_data(self):
        collector = DataCollector(
            paths=[
                os.path.join(self.test_dir, "source1"),
                os.path.join(self.test_dir, "source2"),
            ],
            storage_path=self.storage_path,
            batchsize=3,
        )
        collector.collect_data()

        batch_files = [f for f in os.listdir(
            self.storage_path) if f.endswith(".csv")]
        self.assertEqual(len(batch_files), 2)

        df = pd.read_csv(os.path.join(self.storage_path, "batch_0.csv"))
        self.assertIn("season", df.columns)
        self.assertEqual(df["season"].iloc[0], "April")

    def test_batch_resize(self):
        initial_storage = os.path.join(self.test_dir, "initial_storage")
        os.makedirs(initial_storage, exist_ok=True)

        pd.DataFrame(self.sample_data).to_csv(
            os.path.join(initial_storage, "batch_0.csv"), index=False
        )
        pd.DataFrame(self.sample_data).to_csv(
            os.path.join(initial_storage, "batch_1.csv"), index=False
        )

        collector = DataCollector(
            paths=[],
            storage_path=initial_storage,
            batchsize=3,
        )
        collector.batch_resize()
        batch_files = [f for f in os.listdir(
            initial_storage) if f.endswith(".csv")]
        self.assertEqual(len(batch_files), 2)
        df = pd.read_csv(os.path.join(initial_storage, "batch_0.csv"))
        self.assertEqual(len(df), 3)

        shutil.rmtree(initial_storage)


if __name__ == "__main__":
    unittest.main()
