from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
from typing import List

# Initialize FastAPI
app = FastAPI(title="Image Classification API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Your existing model code (simplified for API) ---
IMG_SIZE = 224

# Load models (you'll need to implement this properly)
def load_models():
    # Place your model loading code here
    pass

# Initialize models
modeli = load_models()  # Your combined model
gbdt_model = None       # Will be loaded with your GBDT model

@app.on_event("startup")
async def startup_event():
    """Initialize models when the app starts"""
    global modeli, gbdt_model
    modeli = load_models()
    # Load your trained GBDT model here
    # gbdt_model = load_gbdt_model()

@app.get("/")
def read_root():
    return {"message": "Image Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess (use your existing function)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Get prediction (replace with your actual prediction logic)
        prediction = "Real"  # Placeholder
        confidence = 0.95    # Placeholder
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}
