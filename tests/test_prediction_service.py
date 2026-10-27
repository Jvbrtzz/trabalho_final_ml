import unittest

import numpy as np

from oncolens.services.prediction_service import PredictionService


class FakeModel:
    def __init__(self, malignant_probability):
        self.malignant_probability = malignant_probability
        self.received_input = None

    def predict_proba(self, model_input):
        self.received_input = model_input
        return np.array([[1 - self.malignant_probability, self.malignant_probability]])


class PredictionServiceTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel(malignant_probability=0.8)
        self.metadata = {
            "api_features": ["first_api_name", "second_api_name"],
            "model_features": ["first model name", "second model name"],
            "class_names": {0: "Benigno", 1: "Maligno"},
            "threshold": 0.5,
        }
        self.service = PredictionService(self.model, self.metadata)

    def test_maps_api_fields_to_model_columns_in_the_expected_order(self):
        result = self.service.predict(
            {"second_api_name": 20.0, "first_api_name": 10.0}
        )

        self.assertEqual(
            list(self.model.received_input.columns),
            ["first model name", "second model name"],
        )
        self.assertEqual(self.model.received_input.iloc[0].tolist(), [10.0, 20.0])
        self.assertEqual(result.predicted_class, "Maligno")
        self.assertEqual(result.malignant_probability, 0.8)

    def test_reports_missing_features(self):
        with self.assertRaisesRegex(ValueError, "second_api_name"):
            self.service.predict({"first_api_name": 10.0})


if __name__ == "__main__":
    unittest.main()
