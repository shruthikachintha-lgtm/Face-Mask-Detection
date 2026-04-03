import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import sys
import os

MODEL_PATH = "mask_detector.h5"
IMG_SIZE   = 224
CLASSES    = ["mask_weared_incorrect", "with_mask", "without_mask"]
COLORS     = {"with_mask": (0, 255, 0), "without_mask": (0, 0, 255), "mask_weared_incorrect": (0, 165, 255)}

# Load model and face detector
print("[INFO] Loading model...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_predict(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)[0]
        idx   = np.argmax(preds)
        label = CLASSES[idx]
        conf  = preds[idx]
        results.append(((x, y, w, h), label, conf))

    return results

# ── Main ─────────────────────────────────────────────────
img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
if not os.path.exists(img_path):
    print(f"[ERROR] Image not found: {img_path}")
    sys.exit(1)

image   = cv2.imread(img_path)
results = detect_and_predict(image)

for ((x, y, w, h), label, conf) in results:
    color = COLORS.get(label, (255, 255, 255))
    text  = f"{label}: {conf*100:.1f}%"
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()