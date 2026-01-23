from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os, json, time, shutil
from multiprocessing import Process

from training_script import train_ai_advanced, predict_excel_process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
STATE_FILE = os.path.join(BASE_DIR, 'progress_state.json')

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def save_state(is_running, progress, message):
    with open(STATE_FILE, 'w') as f:
        json.dump({
            "is_running": is_running,
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }, f)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"is_running": False}
    return json.load(open(STATE_FILE))

def training_process():
    save_state(True, 1, "Initializing Training...")
    train_ai_advanced(progress_callback=lambda p,m: save_state(True,p,m))
    save_state(False, 100, "Training Completed")

@app.post("/train")
def start_training(file: UploadFile = File(...)):
    if load_state().get("is_running"):
        raise HTTPException(400, "Training already running")

    path = os.path.join(DATASET_DIR, "training_data.xlsx")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    Process(target=training_process).start()
    return {"status": "training started"}

@app.post("/predict")
def predict(file: UploadFile = File(...), report_type: str = Query("daily")):
    path = os.path.join(DATASET_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output, _ = predict_excel_process(path, report_type)
    return FileResponse(output)
