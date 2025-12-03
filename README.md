

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

Données CSV brutes
        │
        ▼
(1) ▶ Prétraitement (1_preprocessing.py)
        │    ↳ Enregistre : X_train, y_train, scaler, encoder, etc.
        ▼
(2) ▶ Entraînement (2_train_model.py)
        │    ↳ Enregistre : modèle, métriques dans MLflow
        ▼
(3) ▶ API (api.py) ← charge les artefacts du modèle
        ▲
        │
(4) ◀ Frontend ← appelle l’API `/predict` pour chaque avis
        ▲
        │
(5) ▶ Tests ← assurent la validité du pipeline
        ▼
(6) ▶ CI/CD GitHub Actions (test.yml, docker-build.yml)
        ▼
(7) ▶ Déploiement (Railway + Vercel)



🔗 Liens du projet
✅ Projet GitHub (code complet, CI/CD, structure MLOps)
📂 Repo GitHub principal :
👉 https://github.com/Adelllllllll/DevOps-project

🌐 Frontend (interface utilisateur)
💻 Interface de détection des avis (hébergée sur Vercel) :
/!\ L'API n'est plus hebergée  donc innutile /!\
👉 https://dev-ops-project-gold.vercel.app/

🛠️ API FastAPI (endpoint de prédiction)
🔧 API (hébergée sur Railway)  /!\ N'est plus hebergée /!\:
👉 https://devops-project-production.up.railway.app/docs

proj : https://railway.com/project/73882206-bf89-45ec-88b3-5627c8ac059d?environmentId=5911fe59-e2af-45cd-a9e1-8fc355352545

Model ML :
mlflow ui
http://127.0.0.1:5000/#/models/review-fraud-detector

🐳 DockerHub (image de l’API)
📦 Image DockerHub :
👉 https://hub.docker.com/r/adelllllllll/avis-fraud-api


Badges :

![CI](https://github.com/Adelllllllll/DevOps-project/actions/workflows/test.yml/badge.svg)
![Docker Build](https://github.com/Adelllllllll/DevOps-project/actions/workflows/docker-build.yml/badge.svg)

[![DockerHub](https://hub.docker.com/repository/docker/adellil/fake-review-api/general)



