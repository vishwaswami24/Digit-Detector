from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import app as digit_app


class StubScaler:
    def transform(self, values):
        return values


class StubModel:
    classes_ = np.arange(10)

    def predict_proba(self, values):
        probabilities = np.zeros((1, 10), dtype=float)
        probabilities[0, 3] = 0.72
        probabilities[0, 8] = 0.18
        probabilities[0, 1] = 0.10
        return probabilities


def make_digit_data_url() -> str:
    image = Image.new("L", (280, 280), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((80, 50, 200, 50), fill=0, width=26)
    draw.line((200, 50, 200, 135), fill=0, width=26)
    draw.line((85, 140, 195, 140), fill=0, width=26)
    draw.line((200, 145, 200, 230), fill=0, width=26)
    draw.line((80, 230, 200, 230), fill=0, width=26)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_feedback_file = digit_app.FEEDBACK_FILE
        self.original_model = digit_app.model
        self.original_scaler = digit_app.scaler
        self.original_model_error = digit_app.model_load_error

        digit_app.FEEDBACK_FILE = Path(self.temp_dir.name) / "feedback.csv"
        digit_app.model = StubModel()
        digit_app.scaler = StubScaler()
        digit_app.model_load_error = None
        digit_app.ensure_feedback_store()

        self.client = digit_app.app.test_client()

    def tearDown(self):
        digit_app.FEEDBACK_FILE = self.original_feedback_file
        digit_app.model = self.original_model
        digit_app.scaler = self.original_scaler
        digit_app.model_load_error = self.original_model_error
        self.temp_dir.cleanup()

    def test_predict_returns_ranked_output(self):
        response = self.client.post("/predict", json={"image": make_digit_data_url()})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["digit"], 3)
        self.assertEqual(payload["top_predictions"][0]["digit"], 3)
        self.assertIn("3", payload["probabilities"])

    def test_predict_rejects_blank_canvas(self):
        blank_image = Image.new("L", (280, 280), color=255)
        buffer = io.BytesIO()
        blank_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        response = self.client.post(
            "/predict",
            json={"image": f"data:image/png;base64,{encoded}"},
        )

        self.assertEqual(response.status_code, 400)

    def test_feedback_persists_correction(self):
        response = self.client.post(
            "/feedback",
            json={
                "image": make_digit_data_url(),
                "predicted_digit": 3,
                "correct_digit": 8,
                "confidence": 0.72,
            },
        )
        payload = response.get_json()
        saved_frame = digit_app.read_feedback_frame()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["was_correction"])
        self.assertEqual(len(saved_frame), 1)
        self.assertEqual(int(saved_frame.iloc[0]["correct_digit"]), 8)

    def test_feedback_stats_reports_aggregate_values(self):
        digit_app.append_feedback_row(
            {
                "timestamp": "2026-03-13T10:00:00+00:00",
                "predicted_digit": 3,
                "correct_digit": 3,
                "image_b64": make_digit_data_url(),
                "confidence": 0.72,
            }
        )
        digit_app.append_feedback_row(
            {
                "timestamp": "2026-03-13T10:01:00+00:00",
                "predicted_digit": 8,
                "correct_digit": 1,
                "image_b64": make_digit_data_url(),
                "confidence": 0.40,
            }
        )

        response = self.client.get("/feedback_stats")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["total_feedback"], 2)
        self.assertEqual(payload["wrong_predictions"], 1)
        self.assertAlmostEqual(payload["agreement_rate"], 0.5)
        self.assertAlmostEqual(payload["average_confidence"], 0.56)


if __name__ == "__main__":
    unittest.main()
