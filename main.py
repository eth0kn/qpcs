from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from training_script import train_ai_advanced, predict_excel_process

app = FastAPI(title="QPCS AI Backend")

# ==============================================================================
# CORS (WAJIB BIAR FRONTEND GA FAIL)
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/train")
def train():
    train_ai_advanced()
    return {"status": "training completed"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(OUTPUT_DIR, f"RESULT_{file.filename}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    predict_excel_process(input_path, output_path)

    return {
        "status": "success",
        "output_file": output_path
    }
