import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

columns = ['unit_id', 'time_cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

print("🔄 Chargement des données...")

df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)

max_life = df.groupby('unit_id')['time_cycle'].max().reset_index()
max_life.columns = ['unit_id', 'max_cycle']

df = df.merge(max_life, on='unit_id', how='left')
df['RUL'] = df['max_cycle'] - df['time_cycle']

drop_labels = ['unit_id', 'time_cycle', 'max_cycle', 'setting1', 'setting2', 'setting3', 'RUL']
X = df.drop(columns=drop_labels)
y = df['RUL']

print(f"📊 Données prêtes : {X.shape[0]} lignes, {X.shape[1]} capteurs.")

print("🧠 Entraînement du modèle XGBoost en cours...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"Modèle entraîné ! Erreur moyenne (RMSE) : {rmse:.2f} cycles")

if not os.path.exists('model'):
    os.makedirs('model')
    
joblib.dump(model, 'model/turbofan_rul_model.pkl')
print("💾 Modèle sauvegardé dans 'model/turbofan_rul_model.pkl'")