\# ✈️ Predictive Maintenance: Turbofan Engine RUL Estimator



!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue)

!\[XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)

!\[FastAPI](https://img.shields.io/badge/API-FastAPI-green)

!\[Status](https://img.shields.io/badge/Status-Production%20Ready-success)



\## 📋 Overview

This project is an \*\*Industrial AI solution\*\* designed to predict the \*\*Remaining Useful Life (RUL)\*\* of aircraft engines (Turbofans) based on time-series sensor data. 



Using the \*\*NASA C-MAPSS dataset\*\*, the system analyzes real-time sensor inputs (Temperature, Pressure, Vibration) to anticipate failures before they happen. This approach allows for \*\*predictive maintenance\*\*, reducing downtime and operational costs.



\## 🛠️ Tech Stack

\* \*\*Data Engineering:\*\* Pandas, NumPy (ETL \& Time-series processing)

\* \*\*Machine Learning:\*\* Scikit-learn, XGBoost (Regression Model)

\* \*\*Backend/API:\*\* FastAPI, Uvicorn, Pydantic

\* \*\*Deployment:\*\* Docker ready (Model serialization with Joblib)



\## 🚀 Key Features

\* \*\*Data Ingestion \& Cleaning:\*\* Automated pipeline to process raw sensor logs.

\* \*\*RUL Calculation:\*\* Transformation of time-cycles into "Time-to-Failure" targets.

\* \*\*High-Performance Model:\*\* XGBoost Regressor optimized for tabular sensor data.

\* \*\*Real-time Inference API:\*\* REST API endpoint to query the model with live data.



\## ⚙️ Installation \& Usage



\### 1. Clone the repository

```bash

git clone \[https://github.com/Christ-Fotso/predictive-maintenance-rul.git](https://github.com/Christ-Fotso/predictive-maintenance-rul.git)

cd predictive-maintenance-rul



\###2. Install dependencies

2\. Install dependencies

It is recommended to use a virtual environment.



```bash

pip install -r requirements.txt



\###3. Train the Model

The script will load the NASA dataset, process features, train the XGBoost model, and save it as a .pkl file.



```bash

python train\_model.py



Output: ✅ Modèle entraîné ! Erreur moyenne (RMSE) : 41.38 cycles



\###4. Run the API Server

Start the FastAPI server to make predictions.



```bash

uvicorn app:app --reload



API Documentation

Once the server is running, go to http://127.0.0.1:8000/docs to test the API interactively.



Endpoint: POST /predict

Input: JSON containing values from 21 sensors.



```json

{

&nbsp; "sensors": \[

&nbsp;   518.67, 641.82, 1589.70, 1400.60, 14.62, 

&nbsp;   21.61, 554.36, 2388.06, 9046.19, 1.30, 

&nbsp;   47.47, 521.66, 2388.02, 8138.62, 8.4195, 

&nbsp;   0.03, 392, 2388, 100.0, 39.06, 23.4190

&nbsp; ]

}





Output: Estimated remaining cycles and status.



```json

{

&nbsp; "RUL\_prediction": 178.5,

&nbsp; "unite": "cycles\_restants",

&nbsp; "statut": "🟢 Nominal"

}





\##Dataset Info

The data is provided by the NASA Ames Research Center (Prognostics Data Repository). It simulates the degradation of turbofan engines under different operating conditions.



Author: Arole DJECHE - Engineering Student in AI \& Data



\*\*\*



\### 💡 Petit bonus visuel



Si tu as fait la capture d'écran du résultat vert ("🟢 Nominal"), renomme l'image en `demo\_result.png`, mets-la dans ton dossier, et ajoute cette ligne juste après le titre `Overview` dans le fichier ci-dessus :



```markdown

!\[Demo Result](demo\_result.png)

