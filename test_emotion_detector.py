import unittest
import os
from model_loader import ModelManager, download_models

class TestEmotionDetector(unittest.TestCase):
    def test_download_models(self):
        # Test if models are downloaded and paths are returned
        age_proto, age_model, gender_proto, gender_model, face_proto, face_model = download_models()
        
        self.assertTrue(os.path.exists(age_proto), 'Age proto file should exist')
        self.assertTrue(os.path.exists(age_model), 'Age model file should exist')
        self.assertTrue(os.path.exists(gender_proto), 'Gender proto file should exist')
        self.assertTrue(os.path.exists(gender_model), 'Gender model file should exist')
        self.assertTrue(os.path.exists(face_proto), 'Face proto file should exist')
        self.assertTrue(os.path.exists(face_model), 'Face model file should exist')

    def test_model_manager_initialization(self):
        # Test if ModelManager initializes correctly
        manager = ModelManager()
        self.assertIsNone(manager.age_net)
        self.assertIsNone(manager.gender_net)
        
        # We can't easily test load_models without downloading, but we can check the method exists
        self.assertTrue(hasattr(manager, 'load_models'))

if __name__ == '__main__':
    unittest.main()
