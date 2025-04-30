# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
from datetime import datetime
import uuid

app = FastAPI()

# Allow CORS (replace "*" with your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock prediction function (replace with your actual model)
def predict_image(img_array):
    return {
        "prediction": "Real",  # Replace with model inference
        "confidence": 0.95
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Read and process image
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype('float32') / 255.0
        
        # Get prediction (replace with your model)
        result = predict_image(img)
        
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))