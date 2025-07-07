import os
import string
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import hstack

# === Chargement des artefacts ===
model = joblib.load("data/processed/model.joblib")
tfidf = joblib.load("data/processed/tfidf_vectorizer.joblib")
scaler = joblib.load("data/processed/rating_scaler.joblib")
encoder = joblib.load("data/processed/category_encoder.joblib")
STOPWORDS = set(joblib.load("nltk_stopwords_en.joblib")) if os.path.exists("nltk_stopwords_en.joblib") else set()

# === Création de l'app FastAPI ===
app = FastAPI(title="Fake Review Detection API")

# === Autoriser les requêtes CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Schéma des données d'entrée ===
class ReviewRequest(BaseModel):
    text: str
    rating: float
    product_category: str

# === Fonction de nettoyage de texte ===
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    if STOPWORDS:
        tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

# === Endpoint de prédiction ===
@app.post("/predict")
def predict_review(data: ReviewRequest):
    clean = clean_text(data.text)
    X_text = tfidf.transform([clean])
    X_rating = scaler.transform([[data.rating]])
    X_category = encoder.transform([[data.product_category]])
    X = hstack([X_text, X_rating, X_category])

    pred = model.predict(X)[0]
    label = "Fake" if pred == 1 else "Genuine"
    return {"prediction": label}
