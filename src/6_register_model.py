import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "review-fraud-detector"

# Récupère la dernière version enregistrée
latest_versions = client.get_latest_versions(model_name)
latest_version = max(int(v.version) for v in latest_versions)

# Crée un alias "production" sur la dernière version
client.set_registered_model_alias(model_name, "production", latest_version)

print(f"✅ Alias 'production' ajouté à la version {latest_version} du modèle '{model_name}'")
