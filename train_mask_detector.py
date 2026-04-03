import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import xml.etree.ElementTree as ET
from imutils import paths
import cv2

# ── Config ──────────────────────────────────────────────
DATASET_PATH   = "archive (1)"
IMAGE_DIR      = os.path.join(DATASET_PATH, "images")
ANNOT_DIR      = os.path.join(DATASET_PATH, "annotations")
MODEL_PATH     = "mask_detector.h5"
INIT_LR        = 1e-4
EPOCHS         = 20
BS             = 32
IMG_SIZE       = 224

# ── Load data from XML annotations ──────────────────────
print("[INFO] Loading images and annotations...")
data   = []
labels = []

for xml_file in sorted(os.listdir(ANNOT_DIR)):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(ANNOT_DIR, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue
    h, w = image.shape[:2]

    for obj in root.findall("object"):
        label = obj.find("name").text  # "with_mask" / "without_mask" / "mask_weared_incorrect"
        bbox  = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Clamp to image bounds
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        face = image[ymin:ymax, xmin:xmax]
        if face.size == 0:
            continue

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = preprocess_input(face)

        data.append(face)
        labels.append(label)

print(f"[INFO] Total samples loaded: {len(data)}")

# ── Encode labels ────────────────────────────────────────
data   = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(f"[INFO] Classes: {lb.classes_}")

# ── Train/test split ─────────────────────────────────────
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# ── Data augmentation ────────────────────────────────────
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ── Build model (MobileNetV2 + custom head) ──────────────
print("[INFO] Building model...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# ── Compile & train ──────────────────────────────────────
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

# ── Evaluate ─────────────────────────────────────────────
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_
))

# ── Save model ───────────────────────────────────────────
print(f"[INFO] Saving model to {MODEL_PATH}...")
model.save(MODEL_PATH)
print("[INFO] Training complete!")

# ── Plot training curves ──────────────────────────────────
N = EPOCHS
plt.figure()
plt.plot(H.history["loss"],     label="train loss")
plt.plot(H.history["val_loss"], label="val loss")
plt.plot(H.history["accuracy"],     label="train acc")
plt.plot(H.history["val_accuracy"], label="val acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("training_plot.png")
print("[INFO] Training plot saved as training_plot.png")