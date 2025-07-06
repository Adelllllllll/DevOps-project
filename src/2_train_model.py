import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

print("🔹 Chargement des données...")
X_train = joblib.load("data/processed/X_train.joblib")
X_test = joblib.load("data/processed/X_test.joblib")
y_train = joblib.load("data/processed/y_train.joblib")
y_test = joblib.load("data/processed/y_test.joblib")

print(f"   -> X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"   -> y_train: {len(y_train)} | y_test: {len(y_test)}")

print("🔹 Entraînement du modèle LogisticRegression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("   -> Entraînement terminé.")

print("🔹 Prédiction sur le jeu de test...")
y_pred = model.predict(X_test)
print("   -> Prédictions réalisées.")

print("🔹 Calcul des métriques...")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")         # Correction ici
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
print(f"   -> Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Précision: {precision:.3f} | Rappel: {recall:.3f}")

print("🔹 Tracking dans MLflow...")
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log du modèle
    mlflow.sklearn.log_model(model, "model")
    
joblib.dump(model, "data/processed/model.joblib")


print(f"✅ Modèle entraîné et loggé dans MLFlow : acc={accuracy:.3f} | f1={f1:.3f}")
