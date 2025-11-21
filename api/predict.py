import json
from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load models ONCE
model_cnn = load_model("modeli.h5")
model_gbdt = joblib.load("gbdt_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_array = np.array(data["image"])   # example

    # CNN prediction
    cnn_out = model_cnn.predict(image_array.reshape(1, 224, 224, 3))

    # GBDT prediction
    gbdt_out = model_gbdt.predict([cnn_out.flatten()])

    return json.dumps({
        "cnn_pred": cnn_out.tolist(),
        "gbdt_pred": gbdt_out.tolist()
    })

# Required for Vercel
def handler(request):
    return app(request)
