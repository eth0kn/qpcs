from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from training_script import train_ai_advanced, predict_excel_process

app = FastAPI(title="QPCS AI Backend")

from threading import Thread

TRAIN_STATUS = {
    "running": False,
    "progress": 0,
    "message": "Idle",
    "done": False,
    "error": None
}

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

def training_worker():
    try:
        TRAIN_STATUS["running"] = True
        TRAIN_STATUS["done"] = False
        TRAIN_STATUS["error"] = None

        def progress_callback(p, m):
            TRAIN_STATUS["progress"] = p
            TRAIN_STATUS["message"] = m

        # 🔴 FUNGSI ASLI TETAP DIPAKAI
        train_ai_advanced(progress_callback=progress_callback)

        TRAIN_STATUS["progress"] = 100
        TRAIN_STATUS["message"] = "Training completed"
        TRAIN_STATUS["done"] = True

    except Exception as e:
        TRAIN_STATUS["error"] = str(e)
        TRAIN_STATUS["done"] = True

    finally:
        TRAIN_STATUS["running"] = False


@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/train/status")
def train_status():
    return TRAIN_STATUS

@app.post("/train")
def train():
    if TRAIN_STATUS["running"]:
        return {"status": "already running"}

    thread = Thread(target=training_worker, daemon=True)
    thread.start()

    return {"status": "training started"}

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
