from fastapi import FastAPI

app = FastAPI()

@app.get("/")  # ← This is critical for Railway's health checks
def read_root():
    return {"message": "Hello World"}

# Your other routes...
@app.get("/predict")
def predict():
    return {"result": "success"}
