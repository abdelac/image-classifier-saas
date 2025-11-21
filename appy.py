from fastapi import FastAPI, UploadFile, File
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load models
cnn_model = tf.keras.models.load_model("modeli.h5")
gbdt_model = joblib.load("gbdt_model.pkl")

IMG_SIZE = 224

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess_image(img_bytes)

    # CNN features
    cnn_features = cnn_model.predict(img, verbose=0).flatten().reshape(1, -1)

    # GBDT decision
    pred = gbdt_model.predict(cnn_features)[0]
    confidence = gbdt_model.predict_proba(cnn_features)[0][pred] * 100

    label = "Real Image" if pred == 1 else "Fake Image"

    return {"prediction": label, "confidence": f"{confidence:.2f}%"}

