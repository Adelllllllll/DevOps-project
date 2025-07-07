import requests

def test_e2e_predict_valid():
    url = "https://devops-project-production.up.railway.app/predict"
    payload = {
        "text": "Excellent product",
        "rating": 5.0,
        "product_category": "Books_5"
    }
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], str)


def test_e2e_invalid_payload():
    url = "https://devops-project-production.up.railway.app/predict"
    response = requests.post(url, json={})
    assert response.status_code == 422

def test_e2e_partial_input():
    url = "https://devops-project-production.up.railway.app/predict"
    payload = {"text": "Ok product"}
    response = requests.post(url, json=payload)
    assert response.status_code == 422
