# backend/Dockerfile

# Image officielle Python slim
FROM python:3.11-slim

# Définir le dossier de travail dans le container
WORKDIR /app

# Copier tous les fichiers du backend dans le container
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir fastapi uvicorn numpy scikit-learn pydantic joblib passlib[bcrypt] python-jose

# Exposer le port sur lequel FastAPI va écouter
EXPOSE 8000

# Lancer l'application avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]