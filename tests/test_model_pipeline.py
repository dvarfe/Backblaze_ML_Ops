import os
import unittest
from disk_analyzer.controller import Controller
from disk_analyzer.view import Viewer
from disk_analyzer.utils.constants import PREPROCESSOR_STORAGE, MODELS_VAULT


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        """Set up the controller and viewer for testing."""
        self.controller = Controller()
        self.viewer = Viewer()
        self.sources_path = [os.path.join('tests', 'Test_Data', 'Sources')]
        self.storage_path = os.path.join('tests', 'Test_Data', 'Collected')
        self.preprocessed_path = os.path.join('tests', 'Test_Data', 'Preprocessed')
        self.model_save_path = os.path.join('tests', 'models', 'default.pkl')
        # Path to the configuration file
        self.config_path = './tests/test_model_config.json'

    def test_pipeline(self):
        """Test the model pipeline: collect, preprocess, fit, and save."""
        # Collect Data
        self.controller.collect_data(paths=self.sources_path, storage_path=self.storage_path)
        # Preprocess data
        self.controller.preprocess_data(storage_path=self.storage_path, preprocessed_path=self.preprocessed_path)

        # Fit the model using the configuration file
        self.controller.fit(model_name='NN', cfg=self.config_path,
                            preprocessed_path=self.preprocessed_path)

        # Save the model
        self.controller.save_model(self.model_save_path)

        self.assertTrue(os.path.exists(self.model_save_path))

        self.controller.load_model(self.model_save_path)

        self.controller.fine_tune(preprocessed_path=self.preprocessed_path)

        self.controller.save_best_model(metric='ci', path='models/best_model.pkl')


if __name__ == '__main__':
    unittest.main()
