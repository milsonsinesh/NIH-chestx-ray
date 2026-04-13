# api/main.py

from fastapi import FastAPI, UploadFile
import torch

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    return {
        "prediction": "Effusion (0.82)",
        "disclaimer": "Not a diagnostic tool"
    }
