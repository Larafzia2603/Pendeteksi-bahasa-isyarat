import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import pandas as pd

# Path
MODEL_PATH = r"D:\GISTES\gesture_model.pkl"
CSV_CLEAN = r"D:\GISTES\DatasetFix_clean.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Setup MediaPipe dan pyttsx3
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()
engine.setProperty('rate', 150)
last_gesture = ""

# Load CSV untuk info fitur (opsional)
df_info = pd.read_csv(CSV_CLEAN)
num_features = df_info.shape[1] - 1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Sesuaikan jumlah fitur
            if len(landmarks) != num_features:
                continue

            landmarks = np.array(landmarks).reshape(1, -1)
            gesture = model.predict(landmarks)[0]

            cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if gesture != last_gesture:
                engine.say(gesture)
                engine.runAndWait()
                last_gesture = gesture

    cv2.imshow("Gesture Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
