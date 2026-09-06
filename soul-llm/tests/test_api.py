"""
Unit Tests for FastAPI Backend Endpoints
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)
        self.assertIn("model_name", data)


if __name__ == "__main__":
    unittest.main()
