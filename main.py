from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import threading
import sys

# Import training script yang baru
try:
    from training_script import train_ai_advanced
except ImportError:
    print("Error: training_script.py tidak ditemukan atau error.")
    sys.exit(1)

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State untuk Progress
TRAINING_STATE = {
    "is_running": False,
    "progress": 0,
    "message": "Idle"
}

DATASET_DIR = 'datasets/'
os.makedirs(DATASET_DIR, exist_ok=True)

# ==============================================================================
# HELPER: THREAD WRAPPER
# ==============================================================================
def run_training_process(enable_cleansing):
    """
    Fungsi ini berjalan di background thread.
    """
    global TRAINING_STATE
    TRAINING_STATE["is_running"] = True
    TRAINING_STATE["progress"] = 0
    TRAINING_STATE["message"] = "Memulai proses..."
    
    try:
        # Callback untuk update progress
        def update_progress(pct, msg):
            TRAINING_STATE["progress"] = pct
            TRAINING_STATE["message"] = msg
        
        # Panggil fungsi training (enable_cleansing diabaikan di dalam fungsi, tapi tetap dipassing)
        train_ai_advanced(enable_cleansing=False, progress_callback=update_progress)
        
    except Exception as e:
        print(f"Training Error: {e}")
        TRAINING_STATE["message"] = f"Error: {str(e)}"
        TRAINING_STATE["progress"] = 0 # Reset atau set error state code
    finally:
        TRAINING_STATE["is_running"] = False

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
def read_root():
    return {"status": "QPCS AI Backend Ready"}

@app.post("/train")
async def start_training(file: UploadFile = File(...), enable_cleansing: bool = Query(False)):
    global TRAINING_STATE
    
    if TRAINING_STATE["is_running"]:
        raise HTTPException(status_code=400, detail="Training sedang berjalan. Mohon tunggu.")
    
    # 1. Simpan File Excel
    file_location = os.path.join(DATASET_DIR, "training_data.xlsx")
    try:
        # Hapus file lama jika ada untuk menghindari conflict permission
        if os.path.exists(file_location):
            os.remove(file_location)
            
        with open(file_location, "wb+") as fo:
            shutil.copyfileobj(file.file, fo)
            
    except PermissionError:
        raise HTTPException(status_code=400, detail="File sedang dibuka oleh program lain. Tutup Excel lalu coba lagi.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal upload file: {str(e)}")

    # 2. Jalankan Training di Background Thread
    # Kita force enable_cleansing=False sesuai request user (RAW DATA)
    thread = threading.Thread(target=run_training_process, args=(False,))
    thread.start()
    
    return {"message": "Training dimulai di background...", "status": "started"}

@app.get("/progress")
def get_progress():
    """
    Endpoint ini akan ditembak oleh index.php nanti di Tahap 2
    """
    return TRAINING_STATE

# (Bagian Endpoint /process untuk prediksi nanti akan kita update di tahap selanjutnya)
# Untuk sekarang, biarkan kode lama (jika ada) atau biarkan kosong jika file main.py anda sebelumnya terpotong.