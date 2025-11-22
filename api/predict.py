import json
import numpy as np
from tensorflow.keras.models import load_model
import joblib


def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {local_path}")

# Replace these URLs with your GitHub release URLs
MODEL_CNN_URL = "https://github.com/abdelac/image-classifier-saas/releases/download/Model/modeli.h5"
MODEL_GBDT_URL = "https://github.com/abdelac/image-classifier-saas/releases/download/Model/fake_detectorj.pkl"

download_file(MODEL_CNN_URL, "modeli.h5")
download_file(MODEL_GBDT_URL, "gbdt_model.pkl")





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
