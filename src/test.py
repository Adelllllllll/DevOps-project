import joblib
from scipy.sparse import hstack

# Chargement artefacts
model = joblib.load("data/processed/model.joblib")
tfidf = joblib.load("data/processed/tfidf_vectorizer.joblib")
scaler = joblib.load("data/processed/rating_scaler.joblib")
encoder = joblib.load("data/processed/category_encoder.joblib")

def predict(text, rating, category):
    from string import punctuation
    text_clean = text.lower().translate(str.maketrans("", "", punctuation))
    X_text = tfidf.transform([text_clean])
    X_rating = scaler.transform([[rating]])
    X_cat = encoder.transform([[category]])
    X = hstack([X_text, X_rating, X_cat])
    pred = model.predict(X)[0]
    label = "Fake" if pred == 1 else "Genuine"
    return label

# 🔎 Quelques tests
samples = [
    ("Great phone, loved it!", 5.0, "Electronics"),
    ("Worst product ever", 1.0, "Books_5"),
    ("meh meh meh", 3.0, "Clothing_5"),
    ("Excellent quality, very happy", 5.0, "Home_and_Kitchen_5"),
    ("terrible scam", 1.0, "Books_5"),
    ("caca", 1.0, "caca")
]

for text, rating, cat in samples:
    print(f"[{cat}] {text} ➝ {predict(text, rating, cat)}")
