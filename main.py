from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import threading
import sys
import json
import time
import asyncio
import functools

try:
    from training_script import train_ai_advanced, predict_excel_process
except ImportError:
    print("Error: training_script.py tidak ditemukan.")
    sys.exit(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
STATE_FILE = os.path.join(BASE_DIR, 'progress_state.json')

os.makedirs(DATASET_DIR, exist_ok=True)

# ==============================================================================
# STATE MANAGER (FILE BASED)
# ==============================================================================
def save_state(is_running, progress, message):
    state = {
        "is_running": is_running,
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"is_running": False, "progress": 0, "message": "Idle"}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"is_running": False, "progress": 0, "message": "Idle"}

# Reset state saat start
save_state(False, 0, "Idle")

# ==============================================================================
# THREAD WRAPPER
# ==============================================================================
def run_training_process(enable_cleansing):
    # Set Start State
    save_state(True, 1, "Inisialisasi System (8 Cores)...")
    
    try:
        def update_progress(pct, msg):
            save_state(True, pct, msg)
        
        train_ai_advanced(enable_cleansing=False, progress_callback=update_progress)
        
        # Set Finish State (tahan sebentar biar frontend sempat baca 100%)
        save_state(False, 100, "Training Selesai")
        
    except Exception as e:
        print(f"Training Error: {e}")
        save_state(False, 0, f"Error: {str(e)}")

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
def read_root():
    return {"status": "QPCS AI Backend Turbo Ready"}

@app.post("/train")
async def start_training(file: UploadFile = File(...), enable_cleansing: bool = Query(False)):
    current_state = load_state()
    if current_state["is_running"]:
        raise HTTPException(status_code=400, detail="Training sedang berjalan.")
    
    file_location = os.path.join(DATASET_DIR, "training_data.xlsx")
    try:
        if os.path.exists(file_location):
            os.remove(file_location)
        with open(file_location, "wb+") as fo:
            shutil.copyfileobj(file.file, fo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload gagal: {str(e)}")

    # Set Running State segera
    save_state(True, 0, "Starting process...")

    thread = threading.Thread(target=run_training_process, args=(False,))
    thread.start()
    
    return {"message": "Training started", "status": "started"}

@app.get("/progress")
def get_progress():
    return load_state()

@app.post("/predict")
async def run_prediction(
    file: UploadFile = File(...), 
    report_type: str = Query("daily"),
    enable_cleansing: bool = Query(False)
):
    # Cek State
    current_state = load_state()
    if current_state["is_running"]:
        raise HTTPException(status_code=400, detail="Server sedang sibuk (Training/Predicting).")

    # Set State Awal
    save_state(True, 0, "Upload selesai. Memulai Analisa...")

    temp_filename = f"temp_predict_{file.filename}"
    temp_path = os.path.join(DATASET_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb+") as fo:
            shutil.copyfileobj(file.file, fo)
            
        def update_predict_progress(pct, msg):
            save_state(True, pct, msg)

        loop = asyncio.get_event_loop()
        
        func = functools.partial(
            predict_excel_process, 
            input_file_path=temp_path, 
            report_type=report_type, 
            progress_callback=update_predict_progress
        )
        
        output_path, error_msg = await loop.run_in_executor(None, func)
        
        if os.path.exists(temp_path): os.remove(temp_path)
            
        if not output_path:
            save_state(False, 0, "Prediction Failed")
            raise HTTPException(status_code=500, detail=error_msg)
            
        save_state(False, 100, "Prediction Complete. Downloading...")
        
        return FileResponse(
            path=output_path, 
            filename=os.path.basename(output_path),
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        save_state(False, 0, f"Error: {str(e)}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))