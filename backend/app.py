# app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
import joblib
import numpy as np

# =======================
# FastAPI app
# =======================
app = FastAPI(title="Disease Prediction API 🚀")

# =======================
# CORS pour le frontend
# =======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# JWT & Password
# =======================
SECRET_KEY = "changeme_pour_prod_123"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# =======================
# DB en mémoire
# =======================
users_db = {}

# Créer un utilisateur test pour login immédiat
test_email = "test@example.com"
test_password = "password123"
# Tronquer password à 72 caractères pour Bcrypt
users_db[test_email] = {
    "email": test_email,
    "hashed_password": pwd_context.hash(test_password[:72])
}
print(f"Test user created: {test_email} / {test_password}")

# =======================
# Charger le modèle
# =======================
EXPECTED_FEATURES = 20
try:
    model = joblib.load("deciseas_prediction.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Model and encoder loaded")
except Exception as e:
    print("Error loading model:", e)

# =======================
# Schemas
# =======================
class PredictionRequest(BaseModel):
    data: list[int]

class Token(BaseModel):
    access_token: str
    token_type: str

# =======================
# Utilitaires
# =======================
def verify_password(plain, hashed):
    return pwd_context.verify(plain[:72], hashed)

def get_user(email):
    return users_db.get(email)

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(email, password):
    user = get_user(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or get_user(email) is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# =======================
# Routes
# =======================
@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Email or password incorrect")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/predict")
def predict(req: PredictionRequest, current_user: str = Depends(get_current_user)):
    if len(req.data) != EXPECTED_FEATURES:
        raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_FEATURES} features")
    try:
        arr = np.array(req.data).reshape(1, -1)
        pred_encoded = model.predict(arr)
        pred_label = label_encoder.inverse_transform(pred_encoded)
        return {"prediction": pred_label[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")