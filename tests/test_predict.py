import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_fake():
    payload = {
        "text": "This is a great product! Best I've ever used.",
        "rating": 5,
        "product_category": "Electronics"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Fake", "Genuine"]

def test_predict_empty():
    payload = {
        "text": "",
        "rating": 3,
        "product_category": "Electronics"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Fake", "Genuine"]

def test_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
