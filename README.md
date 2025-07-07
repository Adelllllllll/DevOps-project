# Fake Review Detection API 🚀

Ce projet est une API capable de détecter si un avis produit est authentique ou généré automatiquement, à l’aide d’un modèle de machine learning.

## 📦 Fonctionnalités

- Nettoyage des données textuelles
- Pipeline ML avec TF-IDF, encodage de catégories et mise à l’échelle des notes
- Entraînement d’un modèle `LogisticRegression`
- API REST FastAPI pour la prédiction
- Frontend HTML/CSS/JS simple pour tester l'API
- Intégration continue avec GitHub Actions
- Suivi des expériences avec MLflow
- Gestion des données avec DVC

## 📁 Structure

```
├── data/
├── src/
├── frontend/
├── tests/
├── .github/workflows/
├── requirements.txt
├── dvc.yaml
```

## 🧪 Exécuter les tests

```bash
pytest
```

## 🐳 Docker

Build et lancement :

```bash
docker build -t fake-review-api .
docker run -p 8000:8000 fake-review-api
```

## 🌐 Accès API

```bash
POST http://localhost:8000/predict
```

Body (JSON) :
```json
{
  "text": "Amazing product!",
  "rating": 5,
  "product_category": "Electronics"
}
```

## 🔍 Source de données

Dataset Kaggle avec 20k vrais avis et 20k générés : OR = Original (humains), CG = Computer-generated.

---
