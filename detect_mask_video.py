import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "mask_detector.h5"
IMG_SIZE   = 224
CLASSES    = ["mask_weared_incorrect", "with_mask", "without_mask"]
COLORS     = {"with_mask": (0, 255, 0), "without_mask": (0, 0, 255), "mask_weared_incorrect": (0, 165, 255)}

print("[INFO] Loading model...")
model    = load_model(MODEL_PATH)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]
        idx   = np.argmax(preds)
        label = CLASSES[idx]
        conf  = preds[idx]

        color = COLORS.get(label, (255, 255, 255))
        text  = f"{label}: {conf*100:.1f}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Mask Detection - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()