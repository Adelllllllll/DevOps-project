import joblib
import numpy as np
from scipy.sparse import hstack
import pytest

def test_model_loaded():
    model = joblib.load("data/processed/model.joblib")
    assert model is not None

def test_vectorizer_transform():
    vectorizer = joblib.load("data/processed/tfidf_vectorizer.joblib")
    text = ["Excellent product, very satisfied"]
    X = vectorizer.transform(text)
    assert X.shape[0] == 1

def test_prediction_output_full_pipeline():
    import pandas as pd

    model = joblib.load("data/processed/model.joblib")
    vectorizer = joblib.load("data/processed/tfidf_vectorizer.joblib")
    scaler = joblib.load("data/processed/rating_scaler.joblib")
    encoder = joblib.load("data/processed/category_encoder.joblib")

    input_data = {
        "text": "Excellent product, very satisfied",
        "category": "Electronics_5",
        "rating": 5.0,
        "verified": True
    }

    text_vector = vectorizer.transform([input_data["text"]])
    category_vector = encoder.transform([[input_data["category"]]])
    rating_scaled = scaler.transform([[input_data["rating"]]])

    # Le modèle n'a pas été entraîné avec `verified`, on ne l'ajoute pas
    full_features = hstack([text_vector, category_vector, rating_scaled])

    prediction = model.predict(full_features)
    assert prediction.shape == (1,)
