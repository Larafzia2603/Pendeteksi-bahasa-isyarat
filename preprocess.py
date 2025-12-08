import pandas as pd

# Path CSV
INPUT_CSV = r"D:\GISTES\DatasetFix.csv"
OUTPUT_CSV = r"D:\GISTES\DatasetFix_clean.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# ================== PREPROCESSING ==================
# Contoh preprocessing sederhana: normalisasi nilai fitur ke 0-1
for col in df.columns[:-1]:  # kecuali label
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Simpan hasil preprocessing
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Preprocessing selesai, file tersimpan di {OUTPUT_CSV}")
print("\n📊 Preview 5 data terakhir:")
print(df.tail())
