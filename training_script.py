import pandas as pd
import joblib
import os
import numpy as np
import re
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_PATH = 'datasets/training_data.xlsx'
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' 

# --- KONFIGURASI NAMA SHEET BARU ---
SHEET_DAILY = "PROCESS (DEFECT)"      # Header biasanya di row 2 (index 2)
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"  # Header biasanya di row 1 (index 1)

# ==============================================================================
# AI EMBEDDER (MUST MATCH main.py EXACTLY)
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
# HELPER FUNCTIONS
# ==============================================================================
def clean_text_deep(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'\b(pend_reason|tech_remark|asc_remark|pending|remark)\b', ' ', text)
    text = re.sub(r'\b(set ok|job done|replaced)\b', ' ', text)
    text = re.sub(r'[\:\-\_\|\=\[\]]', ' ', text)
    text = re.sub(r'\b(?=\w*\d)(?=\w*[a-z])\w{7,}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_training_row(text):
    s = str(text).strip()
    blacklist = ["", "nan", "0", "-", "null", "none", "0.0", "."]
    if s.lower() in blacklist: return False
    if len(s) < 3: return False
    if not re.search(r'[a-zA-Z]', s): return False
    return True

# ==============================================================================
# MAIN TRAINING LOGIC (SMART SHEET DETECTION)
# ==============================================================================
def train_ai_advanced(enable_cleansing=True, progress_callback=None):
    def report(p, msg):
        if progress_callback: progress_callback(p, msg)
        print(f"[{p}%] {msg}")

    mode_msg = "ON (Deep Clean)" if enable_cleansing else "OFF (Raw Data)"
    report(5, f"Initializing Training (Cleansing: {mode_msg})...")

    if not os.path.exists(DATASET_PATH):
        report(0, "Error: Dataset file not found.")
        return

    # --- STEP 1: INSPECT SHEETS (SMART CHECK) ---
    report(10, "Inspecting Excel Sheets...")
    try:
        xls = pd.ExcelFile(DATASET_PATH, engine='openpyxl')
        available_sheets = xls.sheet_names
        report(15, f"Found Sheets: {available_sheets}")
    except Exception as e:
        report(0, f"Error reading Excel file: {str(e)}")
        return

    # Flags untuk menentukan apa yang akan ditraining
    has_daily = SHEET_DAILY in available_sheets
    has_monthly = SHEET_MONTHLY in available_sheets
    
    df_daily = None
    df_monthly = None

    # --- STEP 2: LOAD DATA BASED ON AVAILABILITY ---
    report(20, "Loading relevant data...")
    
    try:
        if has_daily:
            # Header=2 karena biasanya ada 2 baris judul di atas
            df_daily = pd.read_excel(xls, sheet_name=SHEET_DAILY, header=2)
            report(25, f"-> Loaded Daily Data: {len(df_daily)} rows")
        
        if has_monthly:
            # Header=1 karena biasanya ada 1 baris judul di atas
            df_monthly = pd.read_excel(xls, sheet_name=SHEET_MONTHLY, header=1)
            report(25, f"-> Loaded Monthly Data: {len(df_monthly)} rows")

    except Exception as e:
        report(0, f"Error loading sheet data: {str(e)}")
        return

    # --- STEP 3: PREPARE DATASETS ---
    input_col = 'PROC_DETAIL_E'
    rf_params = {'n_estimators': 200, 'n_jobs': -1, 'random_state': 42}
    
    # Fungsi Helper Pembersih DataFrame
    def process_dataframe(df):
        df[input_col] = df[input_col].fillna("").astype(str)
        if enable_cleansing:
            df['text_ready'] = df[input_col].apply(clean_text_deep)
            df['valid'] = df['text_ready'].apply(is_valid_training_row)
        else:
            df['text_ready'] = df[input_col]
            df['valid'] = True
        return df[df['valid'] == True].copy().fillna('unknown')

    # --- LOGIC BRANCHING (INTI PERUBAHAN) ---
    
    # Skenario A: Training DEFECT Model (Jalan jika Daily ada ATAU Keduanya ada)
    # Catatan: Jika hanya Monthly, user minta "hanya training monthly", jadi Defect skip.
    run_defect_training = False
    df_defect_final = pd.DataFrame()

    if has_daily and has_monthly:
        report(30, "Mode: HYBRID (Daily + Monthly found). Merging data for Defect Model.")
        cols_defect = [input_col, 'Defect1', 'Defect2', 'Defect3']
        # Gabung data agar model Defect makin pintar
        df_defect_final = pd.concat([df_daily[cols_defect], df_monthly[cols_defect]], ignore_index=True)
        run_defect_training = True
        
    elif has_daily:
        report(30, "Mode: DAILY ONLY. Training Defect Model only.")
        cols_defect = [input_col, 'Defect1', 'Defect2', 'Defect3']
        df_defect_final = df_daily[cols_defect].copy()
        run_defect_training = True
        
    # Skenario B: Training OZ/Category Model (Jalan jika Monthly ada)
    run_oz_training = False
    df_oz_final = pd.DataFrame()

    if has_monthly:
        if not has_daily: report(30, "Mode: MONTHLY ONLY. Training Category Model only.")
        cols_cat = [input_col, 'SVC TYPE', 'DETAIL REASON']
        df_oz_final = df_monthly[cols_cat].copy()
        run_oz_training = True

    # --- STEP 4: EXECUTE TRAINING ---
    
    # 1. TRAIN DEFECT MODEL
    if run_defect_training:
        report(40, "Processing Data for Defect Model...")
        df_defect_ready = process_dataframe(df_defect_final)
        
        if len(df_defect_ready) > 0:
            report(50, f"Training Defect Model ({len(df_defect_ready)} samples)...")
            pipe_defect = Pipeline([
                ('embedder', BertEmbedder(MODEL_NAME)), 
                ('clf', MultiOutputClassifier(RandomForestClassifier(**rf_params)))
            ])
            pipe_defect.fit(df_defect_ready['text_ready'], df_defect_ready[['Defect1', 'Defect2', 'Defect3']])
            
            report(60, "Saving Defect Model...")
            joblib.dump(pipe_defect, f'{MODEL_DIR}model_defect.pkl')
        else:
            report(50, "Warning: No valid data for Defect Model. Skipping.")
    else:
        report(40, "Skipping Defect Model Training (Daily sheet not found).")

    # 2. TRAIN OZ MODEL
    if run_oz_training:
        report(70, "Processing Data for Category/OZ Model...")
        df_oz_ready = process_dataframe(df_oz_final)
        
        if len(df_oz_ready) > 0:
            report(80, f"Training Category Model ({len(df_oz_ready)} samples)...")
            pipe_oz = Pipeline([
                ('embedder', BertEmbedder(MODEL_NAME)), 
                ('clf', MultiOutputClassifier(RandomForestClassifier(**rf_params)))
            ])
            pipe_oz.fit(df_oz_ready['text_ready'], df_oz_ready[['SVC TYPE', 'DETAIL REASON']])
            
            report(90, "Saving Category Model...")
            joblib.dump(pipe_oz, f'{MODEL_DIR}model_oz.pkl')
        else:
            report(80, "Warning: No valid data for Category Model. Skipping.")
    else:
        report(70, "Skipping Category Model Training (Monthly sheet not found).")

    # --- FINISH ---
    report(100, "Process Complete.")

# ==============================================================================
# CLI HANDLER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.set_defaults(clean=True)
    args = parser.parse_args()
    
    train_ai_advanced(enable_cleansing=args.clean)