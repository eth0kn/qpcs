import pandas as pd
import joblib
import os
import numpy as np
import re
import argparse
import gc
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

MODEL_NAME = 'BAAI/bge-m3' 

# CONFIG SHEET
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"
HEADER_INDEX = 1 

# ==============================================================================
# AI EMBEDDER (PERFORMANCE OPTIMIZED FOR 16GB RAM)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"   ...Loading Model: {self.model_name}...")
        # Load model ke CPU (RAM)
        model = SentenceTransformer(self.model_name, device='cpu')
        
        if hasattr(X, 'tolist'): sentences = X.tolist()
        else: sentences = X
        
        # INCREASED BATCH SIZE: 64 (Safe for 16GB RAM)
        print(f"   ...Encoding {len(sentences)} rows with Batch Size 64...")
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
        
        # Clean up model to free RAM for RandomForest
        del model
        gc.collect()
        
        return embeddings

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
    # Filter lebih longgar agar data tidak banyak terbuang
    if not s or s.lower() in ["nan", "null"]: return False
    if len(s) < 2: return False 
    return True

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================
def train_ai_advanced(enable_cleansing=True, progress_callback=None):
    def report(p, msg):
        if progress_callback: progress_callback(p, msg)
        print(f"[{p}%] {msg}")

    gc.collect()
    mode_msg = "ON (Deep Clean)" if enable_cleansing else "OFF (Raw Data)"
    report(5, f"Initializing Training (High Perf Mode)...")

    if not os.path.exists(DATASET_PATH):
        report(0, "Error: Dataset file not found.")
        return

    # --- STEP 1: INSPECT SHEETS ---
    try:
        xls = pd.ExcelFile(DATASET_PATH, engine='openpyxl')
        available_sheets = xls.sheet_names
    except Exception as e:
        report(0, f"Error reading Excel: {str(e)}")
        return

    has_daily = SHEET_DAILY in available_sheets
    has_monthly = SHEET_MONTHLY in available_sheets
    
    # --- STEP 2: LOAD DATA ---
    report(10, "Loading Dataset...")
    df_daily = None
    df_monthly = None

    try:
        if has_daily:
            df_daily = pd.read_excel(xls, sheet_name=SHEET_DAILY, header=HEADER_INDEX)
            df_daily.columns = df_daily.columns.str.strip()
        
        if has_monthly:
            df_monthly = pd.read_excel(xls, sheet_name=SHEET_MONTHLY, header=HEADER_INDEX)
            df_monthly.columns = df_monthly.columns.str.strip()
            
        del xls
        gc.collect()
    except Exception as e:
        report(0, f"Error loading data: {str(e)}")
        return

    # --- STEP 3: PREPARE DATA ---
    input_col = 'PROC_DETAIL_E'
    
    # PERFORMANCE TUNING:
    # n_jobs=4 (Gunakan 4 Core dari 8 Core yang ada) -> Lebih Cepat
    # max_depth=None (Biarkan pohon tumbuh maksimal untuk akurasi lebih tinggi)
    # n_estimators=200 (Jumlah pohon cukup banyak)
    rf_params = {'n_estimators': 200, 'n_jobs': 4, 'random_state': 42}
    
    def process_dataframe(df):
        df[input_col] = df[input_col].fillna("").astype(str)
        if enable_cleansing:
            df['text_ready'] = df[input_col].apply(clean_text_deep)
            df['valid'] = df['text_ready'].apply(is_valid_training_row)
        else:
            df['text_ready'] = df[input_col]
            df['valid'] = True
        return df[df['valid'] == True].copy().fillna('unknown')

    # --- LOGIC BRANCHING ---
    df_defect_final = pd.DataFrame()
    run_defect = False
    
    if has_daily and has_monthly:
        cols = [input_col, 'Defect1', 'Defect2', 'Defect3']
        try:
            df_defect_final = pd.concat([df_daily[cols], df_monthly[cols]], ignore_index=True)
            run_defect = True
        except: pass
    elif has_daily:
        cols = [input_col, 'Defect1', 'Defect2', 'Defect3']
        try:
            df_defect_final = df_daily[cols].copy()
            run_defect = True
        except: pass
        
    if df_daily is not None: del df_daily
    gc.collect()

    df_oz_final = pd.DataFrame()
    run_oz = False
    if has_monthly:
        cols = [input_col, 'SVC TYPE', 'DETAIL REASON']
        try:
            df_oz_final = df_monthly[cols].copy()
            run_oz = True
        except: pass
        
    if df_monthly is not None: del df_monthly
    gc.collect()

    # --- STEP 4: EXECUTE TRAINING ---
    
    # 1. TRAIN DEFECT MODEL
    if run_defect:
        report(20, "Processing Defect Data...")
        df_ready = process_dataframe(df_defect_final)
        del df_defect_final
        gc.collect()
        
        if len(df_ready) > 0:
            report(30, f"Encoding {len(df_ready)} rows...")
            
            embedder = BertEmbedder(MODEL_NAME)
            X_encoded = embedder.transform(df_ready['text_ready'].tolist())
            
            report(50, "Training RandomForest (Defect) [Multi-Core]...")
            clf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
            clf.fit(X_encoded, df_ready[['Defect1', 'Defect2', 'Defect3']])
            
            final_pipe = Pipeline([
                ('embedder', BertEmbedder(MODEL_NAME)), 
                ('clf', clf)
            ])
            
            report(60, "Saving Defect Model...")
            joblib.dump(final_pipe, f'{MODEL_DIR}model_defect.pkl')
            
            del df_ready, X_encoded, clf, final_pipe
            gc.collect()
        else:
            report(50, "No data for Defect Model.")

    # 2. TRAIN OZ MODEL
    if run_oz:
        report(70, "Processing Category Data...")
        df_ready = process_dataframe(df_oz_final)
        del df_oz_final
        gc.collect()
        
        if len(df_ready) > 0:
            report(80, f"Encoding {len(df_ready)} rows...")
            
            embedder = BertEmbedder(MODEL_NAME)
            X_encoded = embedder.transform(df_ready['text_ready'].tolist())
            
            report(90, "Training RandomForest (Category) [Multi-Core]...")
            clf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
            clf.fit(X_encoded, df_ready[['SVC TYPE', 'DETAIL REASON']])
            
            final_pipe = Pipeline([
                ('embedder', BertEmbedder(MODEL_NAME)), 
                ('clf', clf)
            ])
            
            report(95, "Saving Category Model...")
            joblib.dump(final_pipe, f'{MODEL_DIR}model_oz.pkl')
            
            del df_ready, X_encoded, clf, final_pipe
            gc.collect()
        else:
            report(80, "No data for Category Model.")

    report(100, "Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--no-clean', dest='clean', action='store_false')
    parser.set_defaults(clean=True)
    args = parser.parse_args()
    
    train_ai_advanced(enable_cleansing=args.clean)