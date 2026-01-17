from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
import os
import gc
import shutil
import numpy as np
import re
import threading # PENTING: Untuk menjalankan training di background
import time

# --- AI & NLP LIBRARIES ---
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. AI CLASS DEFINITION
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
    def fit(self, X, y=None):
        if self.model is None: self.model = SentenceTransformer(self.model_name)
        return self
    def transform(self, X):
        if self.model is None: self.model = SentenceTransformer(self.model_name)
        if hasattr(X, 'tolist'): sentences = X.tolist()
        else: sentences = X
        return self.model.encode(sentences, show_progress_bar=False)

# ==============================================================================
# 2. SERVER CONFIGURATION
# ==============================================================================
app = FastAPI(title="QPCS AI System API", version="9.0 (Training UI Integrated)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = 'models/'
DATASET_DIR = 'datasets/'
MODEL_DEFECT_PATH = os.path.join(MODEL_DIR, 'model_defect.pkl')
MODEL_OZ_PATH = os.path.join(MODEL_DIR, 'model_oz.pkl')

ai_models = {}

# --- GLOBAL TRAINING STATE (The Memory) ---
TRAINING_STATE = {
    "is_running": False,
    "progress": 0,
    "message": "Idle"
}

# ==============================================================================
# 3. LOADER & HELPER
# ==============================================================================
def load_system_resources():
    print("🚀 [SYSTEM START] Loading AI Models into RAM...")
    try:
        if os.path.exists(MODEL_DEFECT_PATH) and os.path.exists(MODEL_OZ_PATH):
            ai_models['defect'] = joblib.load(MODEL_DEFECT_PATH)
            ai_models['oz'] = joblib.load(MODEL_OZ_PATH)
            print("   ✅ AI Models: LOADED")
        else:
            print("   ⚠️ AI Models: NOT FOUND.")
    except Exception as e:
        print(f"   ❌ AI Models Error: {e}")

load_system_resources()

# ... (Include clean_text_deep and is_valid_text helpers from previous main.py here) ...
# (Saya singkat agar tidak terlalu panjang, tapi pastikan fungsi clean_text_deep & is_valid_text ada disini)
def clean_text_deep(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'\b(pend_reason|tech_remark|asc_remark|pending|remark)\b', ' ', text)
    text = re.sub(r'\b(set ok|job done|replaced)\b', ' ', text)
    text = re.sub(r'[\:\-\_\|\=\[\]]', ' ', text)
    text = re.sub(r'\b(?=\w*\d)(?=\w*[a-z])\w{7,}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_text(text):
    s = str(text).strip()
    blacklist = ["", "nan", "0", "-", "null", "none", "0.0", "."]
    if s.lower() in blacklist: return False
    if len(s) < 3: return False
    if not re.search(r'[a-zA-Z]', s): return False
    return True

def filter_output_columns(df, input_col_name):
    # (Sama seperti script sebelumnya)
    required_cols = ['DATA_TYPE', 'RCPT_NO_ORD_NO', 'CLOSE_DT_RTN_DT', 'SALES_MODEL_SUFFIX', 'SERIAL_NO', 'PARTS_DESC1', 'PARTS_DESC2', 'PARTS_DESC3', 'PROC_DETAIL_E', 'ASC_REMARK_E']
    df_clean = pd.DataFrame()
    for col in required_cols:
        if col in df.columns: df_clean[col] = df[col]
        else: df_clean[col] = "" 
    if 'PROC_DETAIL_E' not in df.columns and input_col_name in df.columns:
        df_clean['PROC_DETAIL_E'] = df[input_col_name]
    return df_clean

# ==============================================================================
# 4. PREDICTION ENDPOINT (STANDARD)
# ==============================================================================
@app.post("/predict")
async def predict_excel(
    file: UploadFile = File(...),
    report_type: str = Query(..., description="Select: 'daily' or 'monthly'"),
    enable_cleansing: bool = Query(True, description="Enable Deep Data Cleansing?")
):
    # ... (Gunakan kode logic /predict dari main.py Versi 8.0 sebelumnya) ...
    # Saya skip bagian ini agar tidak terlalu panjang, tapi pastikan kode /predict V8.0 ada disini.
    # LOGIC NYA SAMA PERSIS.
    if not file.filename.endswith(('.xlsx', '.xls')): raise HTTPException(400, "Invalid file.")
    try:
        contents = await file.read()
        df_raw = pd.read_excel(io.BytesIO(contents), engine='openpyxl')
        input_col = 'PROC_DETAIL_E' # Simplified detection
        if input_col not in df_raw.columns:
             candidates = [c for c in df_raw.columns if 'detail' in str(c).lower()]
             if candidates: input_col = candidates[0]
        
        df_final = filter_output_columns(df_raw, input_col)
        X_temp = df_final['PROC_DETAIL_E'].fillna("").astype(str)
        if enable_cleansing: X_temp = X_temp.apply(clean_text_deep)
        valid_mask = X_temp.apply(is_valid_text)
        X_pred = X_temp[valid_mask].tolist()

        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            if report_type == 'daily':
                df_final['Defect1'] = "-"
                df_final['Defect2'] = "-"
                df_final['Defect3'] = "-"
                if len(X_pred) > 0:
                     pred_def = ai_models['defect'].predict(X_pred)
                     df_final.loc[valid_mask, 'Defect1'] = pred_def[:, 0]
                     df_final.loc[valid_mask, 'Defect2'] = pred_def[:, 1]
                     df_final.loc[valid_mask, 'Defect3'] = pred_def[:, 2]
                df_final.to_excel(writer, sheet_name='Defect Classification', index=False)
            
            elif report_type == 'monthly':
                df_final['Defect1'] = "-"
                df_final['Defect2'] = "-"
                df_final['Defect3'] = "-"
                df_final['SVC TYPE'] = "-"
                df_final['DETAIL REASON'] = "-"
                if len(X_pred) > 0:
                    pred_def = ai_models['defect'].predict(X_pred)
                    df_final.loc[valid_mask, 'Defect1'] = pred_def[:, 0]
                    df_final.loc[valid_mask, 'Defect2'] = pred_def[:, 1]
                    df_final.loc[valid_mask, 'Defect3'] = pred_def[:, 2]
                    pred_oz = ai_models['oz'].predict(X_pred)
                    df_final.loc[valid_mask, 'SVC TYPE'] = pred_oz[:, 0]
                    df_final.loc[valid_mask, 'DETAIL REASON'] = pred_oz[:, 1]
                df_final.to_excel(writer, sheet_name='OZ,MS,IH CATEGORY', index=False)
        
        output_buffer.seek(0)
        return StreamingResponse(output_buffer, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={"Content-Disposition": f"attachment; filename=RESULT_{file.filename}"})
    except Exception as e:
        raise HTTPException(500, f"Server Error: {str(e)}")

# ==============================================================================
# 5. TRAINING ENDPOINTS (NEW!!)
# ==============================================================================

def run_training_process(clean_bool):
    """
    Function runs in a separate thread.
    Updates global TRAINING_STATE variable.
    """
    global TRAINING_STATE
    TRAINING_STATE["is_running"] = True
    TRAINING_STATE["progress"] = 0
    TRAINING_STATE["message"] = "Initializing..."

    try:
        from training_script import train_ai_advanced
        
        # Define callback to update global state
        def update_progress(pct, msg):
            TRAINING_STATE["progress"] = pct
            TRAINING_STATE["message"] = msg
        
        # Run actual training
        train_ai_advanced(enable_cleansing=clean_bool, progress_callback=update_progress)
        
        # Reload models in main thread context
        load_system_resources()
        
    except Exception as e:
        TRAINING_STATE["message"] = f"Error: {str(e)}"
        TRAINING_STATE["progress"] = 0 # Reset on fail
    
    finally:
        TRAINING_STATE["is_running"] = False

@app.post("/train")
async def start_training(
    file: UploadFile = File(...),
    enable_cleansing: bool = Query(True)
):
    global TRAINING_STATE
    if TRAINING_STATE["is_running"]:
        raise HTTPException(400, "Training is already in progress. Please wait.")

    # 1. Save File
    os.makedirs(DATASET_DIR, exist_ok=True)
    file_location = os.path.join(DATASET_DIR, "training_data.xlsx")
    with open(file_location, "wb+") as fo: shutil.copyfileobj(file.file, fo)

    # 2. Start Thread
    thread = threading.Thread(target=run_training_process, args=(enable_cleansing,))
    thread.start()

    return {"status": "started", "message": "Training process initiated."}

@app.get("/train/status")
async def get_training_status():
    return TRAINING_STATE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)