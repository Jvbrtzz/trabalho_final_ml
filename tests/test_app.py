import unittest

from fastapi.testclient import TestClient

from app import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_case = {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "mean_smoothness": 0.1184,
            "mean_compactness": 0.2776,
            "mean_concavity": 0.3001,
            "mean_concave_points": 0.1471,
        }

    def test_health(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["versao"], "2.0.0")

    def test_valid_prediction(self):
        response = self.client.post("/api/v1/predictions", json=self.valid_case)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["classe"], {"Benigno", "Maligno"})
        self.assertGreaterEqual(body["probabilidade_maligno"], 0)
        self.assertLessEqual(body["probabilidade_maligno"], 1)

    def test_out_of_range_input(self):
        invalid = dict(self.valid_case)
        invalid["mean_radius"] = 99
        response = self.client.post("/api/v1/predictions", json=invalid)
        self.assertEqual(response.status_code, 422)

    def test_unknown_input_is_rejected(self):
        invalid = dict(self.valid_case)
        invalid["unknown_feature"] = 123
        response = self.client.post("/api/v1/predictions", json=invalid)
        self.assertEqual(response.status_code, 422)

    def test_home_and_static_assets(self):
        home = self.client.get("/")
        stylesheet = self.client.get("/static/css/styles.css")
        javascript = self.client.get("/static/js/app.js")
        self.assertEqual(home.status_code, 200)
        self.assertIn("OncoLens", home.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(javascript.status_code, 200)


if __name__ == "__main__":
    unittest.main()

