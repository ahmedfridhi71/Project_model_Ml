# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copier tout le code backend
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    scikit-learn \
    pydantic \
    joblib \
    "passlib[bcrypt]==1.7.4" \
    "bcrypt==4.0.1" \
    python-jose \
    python-multipart

EXPOSE 8000

# Lancer l'application FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]