import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import string

# ================== SETTING ==================
GESTURE_LIST = list(string.ascii_uppercase)  # Semua gesture A-Z
JUMLAH_DATA_PER_GESTURE = 30
PATH_CSV = r"D:\GISTES\DatasetFix.csv"

# ================== HEADER ==================
columns = [f"f{i}" for i in range(1, 43)]
columns.append("label")

# ================== LOAD DATA LAMA ==================
if os.path.exists(PATH_CSV):
    df_lama = pd.read_csv(PATH_CSV)
    data = df_lama.values.tolist()
    print("✅ Data lama ditemukan → akan ditambahkan")
    print("\n📊 Preview 5 data terakhir dari file CSV lama:")
    print(df_lama.tail())
else:
    data = []
    print("✅ File baru akan dibuat")

# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# ================== KAMERA ==================
cap = cv2.VideoCapture(0)

print("\nProgram akan otomatis mengambil data setiap 0.5 detik per gesture")
time.sleep(2)

for gesture in GESTURE_LIST:
    print(f"\n🎯 Mulai ambil data untuk gesture '{gesture}'")
    count = 0
    start_time = time.time()
    
    while count < JUMLAH_DATA_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                # Simpan data otomatis setiap 0.5 detik
                if (time.time() - start_time) > 0.5:
                    data.append(landmarks + [gesture])
                    count += 1
                    start_time = time.time()
                    print(f"✅ {gesture} tersimpan: {count}/{JUMLAH_DATA_PER_GESTURE}")

        cv2.putText(frame, f"Gesture: {gesture} ({count}/{JUMLAH_DATA_PER_GESTURE})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Ambil Dataset Otomatis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Program dihentikan!")
            cap.release()
            cv2.destroyAllWindows()
            exit()

# ================== SIMPAN CSV ==================
df = pd.DataFrame(data, columns=columns)
df.to_csv(PATH_CSV, index=False)

print("\nSemua data berhasil disimpan di:", PATH_CSV)
print("\nPreview 5 data terakhir setelah input baru:")
print(df.tail())

cap.release()
cv2.destroyAllWindows()
