import json
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load models
model_cnn = load_model("modeli.h5")
model_gbdt = joblib.load("gbdt_model.pkl")

def handler(request):
    try:
        data = request.get_json()  # get POST JSON
        image_array = np.array(data["image"])

        # CNN prediction
        cnn_out = model_cnn.predict(image_array.reshape(1, 224, 224, 3))

        # GBDT prediction
        gbdt_out = model_gbdt.predict([cnn_out.flatten()])

        return {
            "statusCode": 200,
            "body": json.dumps({
                "cnn_pred": cnn_out.tolist(),
                "gbdt_pred": gbdt_out.tolist()
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
