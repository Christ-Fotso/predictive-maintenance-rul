from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Industrial AI - Predictive Maintenance",
    description="API pour prédire le RUL (Remaining Useful Life) des moteurs d'avion.",
    version="1.0.0"
)

try:
    model = joblib.load('model/turbofan_rul_model.pkl')
    print("✅ Modèle IA chargé avec succès !")
except:
    print("⚠️ Erreur : Modèle introuvable. Vérifie le dossier model/.")

class SensorData(BaseModel):
    sensors: list[float]

@app.get("/")
def home():
    return {"status": "API Online", "usage": "Go to /docs to test predictions"}

@app.post("/predict")
def predict_rul(data: SensorData):
    if len(data.sensors) != 21:
        raise HTTPException(status_code=400, detail="Il faut 21 valeurs de capteurs.")

    features = [f's{i}' for i in range(1, 22)]
    df_input = pd.DataFrame([data.sensors], columns=features)

    prediction = model.predict(df_input)[0]

    etat = "🟢 Nominal"
    if prediction < 50:
        etat = "🔴 Maintenance Urgente"
    elif prediction < 100:
        etat = "🟠 Attention"

    return {
        "RUL_prediction": round(float(prediction), 2),
        "unite": "cycles_restants",
        "statut": etat
    }