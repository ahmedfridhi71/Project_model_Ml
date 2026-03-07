# backend/app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import joblib
import numpy as np
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.middleware.cors import CORSMiddleware

# ===============================
# Initialisation FastAPI
# ===============================
app = FastAPI(title="Disease Prediction API 🚀")

# ===============================
# CORS (autoriser le frontend)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour production : ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Configuration JWT
# ===============================
SECRET_KEY = "changeme_pour_prod_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ===============================
# Utilisateurs fictifs pour test
# ===============================
fake_users_db = {
    "user@example.com": {
        "username": "user",
        "hashed_password": pwd_context.hash("password123")
    },
    "ahmed@example.com": {
        "username": "ahmed",
        "hashed_password": pwd_context.hash("1234")  # mot de passe test
    }
}

# ===============================
# Charger le modèle et l'encodeur
# ===============================
EXPECTED_FEATURES = 20
try:
    model = joblib.load("deciseas_prediction.pkl")
    print("Model loaded")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Label encoder loaded")
except Exception as e:
    print(f"Error loading model or encoder: {e}")

# ===============================
# Pydantic Schemas
# ===============================
class PredictionRequest(BaseModel):
    data: list[int]

class Token(BaseModel):
    access_token: str
    token_type: str

# ===============================
# Fonctions utilitaires
# ===============================
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(email: str, password: str):
    user = fake_users_db.get(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ===============================
# Routes
# ===============================
@app.get("/")
def home():
    return {"message": "Disease Prediction API is running 🚀"}

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Email ou mot de passe incorrect")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict")
def predict(request: PredictionRequest, current_user: str = Depends(get_current_user)):
    if len(request.data) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} values but got {len(request.data)}"
        )
    try:
        input_array = np.array(request.data).reshape(1, -1)
        prediction_encoded = model.predict(input_array)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        return {"prediction": prediction_label[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")