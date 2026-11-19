"""
test_api.py — Integration Tests for Flask API Endpoints
CVRT Engineering College, Tadipatri | JNTUA B.Tech CSE 2024-25
Run: python -m pytest tests/test_api.py -v
"""

import os
import sys
import json
import io
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import create_app
from database import init_database


class TestAPIEndpoints(unittest.TestCase):
    """Integration tests for all REST API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Initialize test Flask app with test database"""
        import tempfile
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'plantguard_test_uploads')
        os.makedirs(cls.app.config['UPLOAD_FOLDER'], exist_ok=True)
        cls.client = cls.app.test_client()
        init_database()
        from database import get_connection
        conn = get_connection()
        conn.cursor().execute("DELETE FROM users WHERE username IN ('testuser1', 'dupuser', 'logintest', 'badpassuser')")
        conn.commit()
        conn.close()

    def create_test_image_bytes(self) -> bytes:
        """Create a minimal valid JPEG image in memory for upload tests"""
        from PIL import Image
        img = Image.new('RGB', (224, 224), color=(100, 160, 80))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        return buf.getvalue()

    # ── Health Check ───────────────────────────────────────────
    def test_TC101_health_check_returns_200(self):
        """TC-101: GET /api/health returns 200 with success=true"""
        res = self.client.get('/api/health')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'healthy')

    # ── Registration ───────────────────────────────────────────
    def test_TC102_register_valid_user(self):
        """TC-102: POST /api/register with valid data returns 201"""
        res = self.client.post('/api/register',
            json={'username': 'testuser1', 'email': 'test1@example.com', 'password': 'password123'})
        data = res.get_json()
        self.assertEqual(res.status_code, 201)
        self.assertTrue(data['success'])
        self.assertIn('user_id', data)

    def test_TC103_register_missing_fields_returns_400(self):
        """TC-103: POST /api/register with missing fields returns 400"""
        res = self.client.post('/api/register', json={'username': 'incomplete'})
        data = res.get_json()
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data['success'])

    def test_TC104_register_duplicate_username_returns_400(self):
        """TC-104: POST /api/register with duplicate username returns 400"""
        self.client.post('/api/register',
            json={'username': 'dupuser', 'email': 'dup@example.com', 'password': 'pass123'})
        res = self.client.post('/api/register',
            json={'username': 'dupuser', 'email': 'dup2@example.com', 'password': 'pass123'})
        data = res.get_json()
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data['success'])

    # ── Login ──────────────────────────────────────────────────
    def test_TC105_login_valid_credentials(self):
        """TC-105: POST /api/login with valid credentials returns token"""
        # Register first
        self.client.post('/api/register',
            json={'username': 'logintest', 'email': 'login@example.com', 'password': 'secret123'})

        res = self.client.post('/api/login',
            json={'username': 'logintest', 'password': 'secret123'})
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('token', data)
        self.assertIsInstance(data['token'], str)
        self.assertGreater(len(data['token']), 10)

    def test_TC106_login_invalid_password_returns_401(self):
        """TC-106: POST /api/login with wrong password returns 401"""
        self.client.post('/api/register',
            json={'username': 'badpassuser', 'email': 'bp@example.com', 'password': 'correct'})
        res = self.client.post('/api/login',
            json={'username': 'badpassuser', 'password': 'wrongpassword'})
        data = res.get_json()
        self.assertEqual(res.status_code, 401)
        self.assertFalse(data['success'])

    def test_TC107_login_nonexistent_user_returns_401(self):
        """TC-107: POST /api/login with non-existent username returns 401"""
        res = self.client.post('/api/login',
            json={'username': 'userThatDoesNotExist999', 'password': 'anything'})
        self.assertEqual(res.status_code, 401)

    # ── Prediction Endpoint ────────────────────────────────────
    def test_TC108_predict_without_image_returns_400(self):
        """TC-108: POST /api/predict with no image file returns 400"""
        res = self.client.post('/api/predict', data={})
        data = res.get_json()
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

    def test_TC109_predict_with_invalid_file_type_returns_400(self):
        """TC-109: POST /api/predict with .txt file returns 400"""
        data_payload = {'image': (io.BytesIO(b'hello world'), 'test.txt')}
        res = self.client.post('/api/predict',
            data=data_payload, content_type='multipart/form-data')
        data = res.get_json()
        self.assertEqual(res.status_code, 400)
        self.assertFalse(data['success'])

    # ── History Endpoint ───────────────────────────────────────
    def test_TC110_history_endpoint_returns_200(self):
        """TC-110: GET /api/history returns 200 with success=true and list"""
        res = self.client.get('/api/history')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)

    def test_TC111_history_limit_parameter(self):
        """TC-111: GET /api/history?limit=5 returns at most 5 records"""
        res = self.client.get('/api/history?limit=5')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertLessEqual(len(data['predictions']), 5)

    # ── Diseases Endpoint ──────────────────────────────────────
    def test_TC112_diseases_endpoint_returns_list(self):
        """TC-112: GET /api/diseases returns a non-empty list of disease classes"""
        res = self.client.get('/api/diseases')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('diseases', data)
        self.assertGreater(len(data['diseases']), 0)

    def test_TC113_disease_objects_have_required_fields(self):
        """TC-113: Each disease object has class_name, display_name, crop, severity"""
        res = self.client.get('/api/diseases')
        data = res.get_json()
        for disease in data['diseases'][:5]:   # check first 5
            self.assertIn('class_name', disease)
            self.assertIn('display_name', disease)
            self.assertIn('crop', disease)
            self.assertIn('severity', disease)

    # ── Stats Endpoint ─────────────────────────────────────────
    def test_TC114_stats_endpoint_returns_counts(self):
        """TC-114: GET /api/stats returns statistics and model info"""
        res = self.client.get('/api/stats')
        data = res.get_json()
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('stats', data)
        self.assertIn('total_predictions', data['stats'])
        self.assertIn('total_users', data['stats'])

    def test_TC115_404_returns_json_error(self):
        """TC-115: GET /api/nonexistent returns JSON 404 error"""
        res = self.client.get('/api/nonexistent_endpoint_xyz')
        data = res.get_json()
        self.assertEqual(res.status_code, 404)
        self.assertFalse(data['success'])


if __name__ == '__main__':
    print("Running Plant Disease Detection — API Integration Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
