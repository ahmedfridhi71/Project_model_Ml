from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# ===============================
# Initialisation FastAPI
# ===============================
app = FastAPI(title="Disease Prediction API 🚀")

# ===============================
# Autoriser le frontend à appeler l'API
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour production : ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Configuration
# ===============================
EXPECTED_FEATURES = 20  # Nombre exact de symptômes

# ===============================
# Charger le modèle et l'encodeur
# ===============================
try:
    model = joblib.load("deciseas_prediction.pkl")
    print("Model loaded")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Label encoder loaded")
except Exception as e:
    print(f"Error loading model or encoder: {e}")

# ===============================
# Pydantic Schema pour JSON
# ===============================
class PredictionRequest(BaseModel):
    data: list[int]  # Liste d'entiers (0 ou 1)

# ===============================
# Routes
# ===============================
@app.get("/")
def home():
    return {"message": "Disease Prediction API is running 🚀"}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Vérifier que la liste contient exactement 20 symptômes
    if len(request.data) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} values but got {len(request.data)}"
        )
    try:
        # Convertir en array numpy
        input_array = np.array(request.data).reshape(1, -1)

        # Prédiction
        prediction_encoded = model.predict(input_array)

        # Décodage label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        return {"prediction": prediction_label[0]}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )