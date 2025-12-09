import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================== PATH ==================
MODEL_PATH = r"D:\GISTES\gesture_model.pkl"
CSV_CLEAN = r"D:\GISTES\DatasetFix_clean.csv"

# ================== LOAD MODEL ==================
model = joblib.load(MODEL_PATH)

# ================== LOAD DATASET ==================
df = pd.read_csv(CSV_CLEAN)
X = df.iloc[:, :-1].values
y_true = df['label'].values

# ================== PREDIKSI ==================
y_pred = model.predict(X)

# ================== EVALUASI ==================
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=4)
precision_matrix = precision_score(y_true, y_pred, average=None, labels=np.unique(y_true))

print("\n================= EVALUASI MODEL =================")
print(f"Accuracy: {accuracy:.4f}\n")
print("Precision per kelas:")
for label, prec in zip(np.unique(y_true), precision_matrix):
    print(f"{label}: {prec:.4f}")
print("\nDetail Precision, Recall, F1-Score per kelas:")
print(report)
print("==================================================")

# ================== CONFUSION MATRIX ==================
cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
