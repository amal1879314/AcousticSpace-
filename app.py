from fastapi import FastAPI, UploadFile, File

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
    return {
        "filename": file.filename,
        "message": "Model not loaded yet."
    }
