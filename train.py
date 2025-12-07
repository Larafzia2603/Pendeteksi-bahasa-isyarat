import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Path CSV
INPUT_CSV = r"D:\GISTES\DatasetFix_clean.csv"
MODEL_PATH = r"D:\GISTES\gesture_model.pkl"

# Load data
df = pd.read_csv(INPUT_CSV)
X = df.iloc[:, :-1].values
y = df['label'].values

# Train model
model = RandomForestClassifier()
model.fit(X, y)
print("✅ Model berhasil dilatih")

# Simpan model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model tersimpan di {MODEL_PATH}")
