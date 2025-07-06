import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import string
import joblib
import os

nltk.download('stopwords')

print("🔹 Chargement du dataset...")
df = pd.read_csv("data/raw/fake_reviews_dataset.csv")

# === Nettoyage texte ===
STOPWORDS = set(stopwords.words('english'))  # Chargement 1 seule fois

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

print("🔹 Nettoyage du texte...")
df["clean_text"] = df["text_"].apply(clean_text)

print("🔹 TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df["clean_text"])

print("🔹 Normalisation du rating...")
scaler = StandardScaler()
X_rating = scaler.fit_transform(df[["rating"]])

print("🔹 Encodage OneHot de la catégorie...")
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_category = encoder.fit_transform(df[["category"]])

print("🔹 Fusion des features...")
X = hstack([X_text, X_rating, X_category])
y = df["label"]

print("🔹 Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔹 Sauvegarde des fichiers...")
os.makedirs("data/processed", exist_ok=True)
joblib.dump(X_train, "data/processed/X_train.joblib")
joblib.dump(X_test, "data/processed/X_test.joblib")
joblib.dump(y_train, "data/processed/y_train.joblib")
joblib.dump(y_test, "data/processed/y_test.joblib")
joblib.dump(tfidf, "data/processed/tfidf_vectorizer.joblib")
joblib.dump(scaler, "data/processed/rating_scaler.joblib")
joblib.dump(encoder, "data/processed/category_encoder.joblib")

print("✅ Prétraitement enrichi terminé.")
