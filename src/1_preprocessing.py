import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# 🔹 Chargement
print("🔹 Chargement des données...")
df = pd.read_csv("data/raw/fake_reviews_dataset.csv")

# 🔹 Nettoyage texte
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df["cleaned_text"] = df["text_"].apply(clean_text)  # <-- corrigé ici
df["label"] = df["label"].map({"OR": 0, "CG": 1})    # <-- encodage binaire

# 🔹 Vectorisation
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df["cleaned_text"])

scaler = StandardScaler()
X_rating = scaler.fit_transform(df[["rating"]])

encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_category = encoder.fit_transform(df[["category"]])

X = hstack([X_text, X_rating, X_category])
y = df["label"]

# 🔹 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Sauvegarde
joblib.dump(X_train, "data/processed/X_train.joblib")
joblib.dump(X_test, "data/processed/X_test.joblib")
joblib.dump(y_train, "data/processed/y_train.joblib")
joblib.dump(y_test, "data/processed/y_test.joblib")
joblib.dump(tfidf, "data/processed/tfidf_vectorizer.joblib")
joblib.dump(scaler, "data/processed/rating_scaler.joblib")
joblib.dump(encoder, "data/processed/category_encoder.joblib")

print("✅ Données prétraitées et sauvegardées.")
