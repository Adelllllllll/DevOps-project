import pytest
from src.api import app
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


client = TestClient(app)

def test_predict_valid_input():
    payload = {
        "reviewText": "This product is amazing! Highly recommended.",
        "summary": "Great product",
        "overall": 5.0,
        "verified": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_valid_input():
    payload = {
        "text": "This product is amazing! Highly recommended.",
        "rating": 5.0,
        "product_category": "Books"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Fake", "Genuine"]


def test_predict_empty_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422
