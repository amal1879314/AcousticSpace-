from fastapi import FastAPI, UploadFile, File
from predict import predict_audio

app = FastAPI(
    title="AcousticSpace API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "AcousticSpace Backend Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": False
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = predict_audio(file)

    return result
