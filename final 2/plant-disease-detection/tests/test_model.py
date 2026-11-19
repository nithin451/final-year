"""
test_model.py — Unit Tests for ML Model
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
Run: python -m pytest tests/test_model.py -v
"""

import os
import sys
import json
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

ML_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml')
LABELS_PATH = os.path.join(ML_DIR, 'class_labels.json')
MODEL_PATH = os.path.join(ML_DIR, 'plant_disease_model.h5')
METRICS_PATH = os.path.join(ML_DIR, 'evaluation_metrics.json')


class TestPreprocessing(unittest.TestCase):
    """TC-001 to TC-005: Test data preprocessing functions"""

    def test_preprocess_image_bytes_valid(self):
        """TC-001: preprocess_image_bytes() returns correct shape for valid image"""
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        # Create a 256x256 test image
        img = Image.new('RGB', (256, 256), color=(100, 150, 80))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        image_bytes = buf.read()

        result = preprocess_image_bytes(image_bytes)
        self.assertEqual(result.shape, (1, 224, 224, 3),
                         "Preprocessed image should have shape (1, 224, 224, 3)")

    def test_preprocess_image_bytes_rgba(self):
        """TC-002: preprocess_image_bytes() handles RGBA PNG (converts to RGB)"""
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGBA', (300, 300), color=(100, 150, 80, 128))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        image_bytes = buf.read()

        result = preprocess_image_bytes(image_bytes)
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_preprocess_image_bytes_invalid_raises(self):
        """TC-003: preprocess_image_bytes() raises ValueError for invalid bytes"""
        from preprocess import preprocess_image_bytes

        with self.assertRaises((ValueError, Exception)):
            preprocess_image_bytes(b'this is not valid image data')

    def test_preprocess_image_values_range(self):
        """TC-004: MobileNetV2 preprocessing returns values in [-1.0, 1.0] range"""
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color=(255, 255, 255))  # white
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        image_bytes = buf.read()

        result = preprocess_image_bytes(image_bytes)
        self.assertGreaterEqual(result.min(), -1.1, "Min value should be >= -1")
        self.assertLessEqual(result.max(), 1.1, "Max value should be <= 1")

    def test_preprocess_single_image_invalid_path(self):
        """TC-005: preprocess_single_image() raises FileNotFoundError for missing file"""
        from preprocess import preprocess_single_image

        with self.assertRaises(FileNotFoundError):
            preprocess_single_image('/nonexistent/path/image.jpg')


class TestClassLabels(unittest.TestCase):
    """TC-006 to TC-008: Test class label loading"""

    def test_class_labels_file_exists(self):
        """TC-006: class_labels.json exists after training"""
        self.assertTrue(
            os.path.isfile(LABELS_PATH),
            "class_labels.json should exist. Run ml/train.py first."
        )

    def test_class_labels_count(self):
        """TC-007: class_labels.json contains exactly 38 classes for PlantVillage"""
        if not os.path.isfile(LABELS_PATH):
            self.skipTest("class_labels.json not found — run ml/train.py first")

        with open(LABELS_PATH) as f:
            labels = json.load(f)
        self.assertEqual(len(labels), 38,
                         f"Expected 38 classes, found {len(labels)}")

    def test_class_labels_valid_format(self):
        """TC-008: Each label is a non-empty string containing crop and disease info"""
        if not os.path.isfile(LABELS_PATH):
            self.skipTest("class_labels.json not found — run ml/train.py first")

        with open(LABELS_PATH) as f:
            labels = json.load(f)

        for idx, name in labels.items():
            self.assertIsInstance(name, str, f"Label at index {idx} must be a string")
            self.assertGreater(len(name), 0, f"Label at index {idx} must not be empty")


class TestModelPrediction(unittest.TestCase):
    """TC-009 to TC-015: Test model prediction with the model.py wrapper"""

    def setUp(self):
        """Skip all tests in this class if model file doesn't exist"""
        if not os.path.isfile(MODEL_PATH):
            self.skipTest("Model file not found — run ml/train.py first")

    def test_predict_returns_dict(self):
        """TC-009: predict() returns a dict with required keys"""
        from model import predict
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color=(80, 120, 60))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_array = preprocess_image_bytes(buf.getvalue())

        result = predict(img_array)
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('top3', result)

    def test_predict_confidence_range(self):
        """TC-010: Confidence score is a float between 0 and 1"""
        from model import predict
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color=(80, 120, 60))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_array = preprocess_image_bytes(buf.getvalue())

        result = predict(img_array)
        confidence = result['confidence']
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_predict_top3_length(self):
        """TC-011: Top-3 predictions list contains exactly 3 items"""
        from model import predict
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color=(80, 120, 60))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_array = preprocess_image_bytes(buf.getvalue())

        result = predict(img_array)
        self.assertEqual(len(result['top3']), 3)

    def test_predict_class_in_labels(self):
        """TC-012: Predicted class name exists in the loaded class labels"""
        from model import predict, load_model
        from preprocess import preprocess_image_bytes
        from PIL import Image
        import io

        img = Image.new('RGB', (224, 224), color=(80, 120, 60))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_array = preprocess_image_bytes(buf.getvalue())

        result = predict(img_array)
        _, class_labels = load_model()
        all_labels = list(class_labels.values())
        self.assertIn(result['predicted_class'], all_labels)

    def test_model_loads_once(self):
        """TC-013: Calling load_model() twice returns the same model instance (singleton)"""
        from model import load_model

        model1, labels1 = load_model()
        model2, labels2 = load_model()
        self.assertIs(model1, model2, "Model should be loaded only once (singleton pattern)")


class TestEvaluationMetrics(unittest.TestCase):
    """TC-014 to TC-015: Test saved evaluation metrics"""

    def test_metrics_file_exists(self):
        """TC-014: evaluation_metrics.json exists after running evaluate.py"""
        self.assertTrue(
            os.path.isfile(METRICS_PATH),
            "evaluation_metrics.json not found — run ml/evaluate.py first"
        )

    def test_accuracy_above_threshold(self):
        """TC-015: Saved model accuracy must be >= 0.85 (85%) per project requirements"""
        if not os.path.isfile(METRICS_PATH):
            self.skipTest("evaluation_metrics.json not found — run ml/evaluate.py first")

        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        accuracy = metrics.get('accuracy', 0)
        self.assertGreaterEqual(accuracy, 0.85,
                                f"Accuracy {accuracy:.3f} is below the required 85% threshold")


if __name__ == '__main__':
    print("Running Plant Disease Detection — Model Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
