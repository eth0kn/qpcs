import pandas as pd
import joblib
import os
import numpy as np
import re
import argparse
import gc
import math
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

# NAMA SHEET
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"

# ==============================================================================
# AI EMBEDDER (GRANULAR PROGRESS REPORTING)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, progress_callback=None, start_pct=0, end_pct=0):
        self.model_name = model_name
        self.model = None
        self.progress_callback = progress_callback
        self.start_pct = start_pct # Persentase awal (misal 20%)
        self.end_pct = end_pct     # Persentase akhir (misal 80%)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.progress_callback: 
            self.progress_callback(self.start_pct, f"Loading Model AI ({self.model_name})...")
        
        # Load Model ke CPU
        model = SentenceTransformer(self.model_name, device='cpu')
        
        if hasattr(X, 'tolist'): sentences = X.tolist()
        else: sentences = X
        
        total_sentences = len(sentences)
        batch_size = 64
        all_embeddings = []
        
        # --- MANUAL BATCHING UNTUK GRANULAR PROGRESS ---
        num_batches = math.ceil(total_sentences / batch_size)
        
        print(f"   ...Encoding {total_sentences} rows in {num_batches} batches...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_sentences)
            batch = sentences[start_idx:end_idx]
            
            # Encode batch ini
            batch_emb = model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            
            # Hitung Progress Real-time
            # Kita mapping progress batch (0-100%) ke rentang global (start_pct - end_pct)
            if self.progress_callback:
                batch_progress = (i + 1) / num_batches
                current_global_pct = self.start_pct + int(batch_progress * (self.end_pct - self.start_pct))
                
                # Update UI: "Encoding Batch 5/20..."
                msg = f"Encoding Data: Batch {i+1}/{num_batches}"
                self.progress_callback(current_global_pct, msg)
        
        # Gabungkan semua batch
        final_embeddings = np.vstack(all_embeddings)
        
        # Bersihkan RAM
        del model
        gc.collect()
        
        return final_embeddings

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
    if not s or s.lower() in ["nan", "null"]: return False
    if len(s) < 2: return False 
    return True

def load_sheet_auto_header(xls_file, sheet_name):
    try:
        temp_df = pd.read_excel(xls_file, sheet_name=sheet_name, header=None, nrows=10, engine='openpyxl')
        header_idx = 0
        keywords = ['DATA_TYPE', 'PROC_DETAIL_E', 'SERIAL_NO', 'PARTS_DESC1', 'CLOSE_DT_RTN_DT']
        for idx, row in temp_df.iterrows():
            row_str = [str(val).strip().upper() for val in row.values]
            if sum(1 for k in keywords if k in row_str) >= 2:
                header_idx = idx
                break
        return pd.read_excel(xls_file, sheet_name=sheet_name, header=header_idx, engine='openpyxl')
    except: return None

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================
# DEFAULT CLEANSING SEKARANG FALSE
def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    def report(p, msg):
        if progress_callback: progress_callback(p, msg)
        print(f"[{p}%] {msg}")

    gc.collect()
    mode_msg = "ON" if enable_cleansing else "OFF"
    report(5, f"Init Training (Clean: {mode_msg})...")

    if not os.path.exists(DATASET_PATH):
        report(0, "Error: Dataset file not found.")
        return

    # --- LOAD DATA ---
    try:
        xls = pd.ExcelFile(DATASET_PATH, engine='openpyxl')
        s_names = xls.sheet_names
    except Exception as e:
        report(0, f"Error: {e}")
        return

    report(10, "Loading Sheets...")
    df_daily = load_sheet_auto_header(xls, SHEET_DAILY) if SHEET_DAILY in s_names else None
    df_monthly = load_sheet_auto_header(xls, SHEET_MONTHLY) if SHEET_MONTHLY in s_names else None
    
    if df_daily is not None: df_daily.columns = df_daily.columns.str.strip()
    if df_monthly is not None: df_monthly.columns = df_monthly.columns.str.strip()
    del xls
    gc.collect()

    # --- PREPARE DATA ---
    input_col = 'PROC_DETAIL_E'
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

    # Prepare DataFrames
    df_defect_final = pd.DataFrame()
    run_defect = False
    if df_daily is not None:
        cols = [input_col, 'Defect1', 'Defect2', 'Defect3']
        try:
            # Jika Monthly ada, gabung. Jika tidak, pakai Daily saja.
            if df_monthly is not None:
                df_defect_final = pd.concat([df_daily[cols], df_monthly[cols]], ignore_index=True)
            else:
                df_defect_final = df_daily[cols].copy()
            run_defect = True
        except: pass

    df_oz_final = pd.DataFrame()
    run_oz = False
    if df_monthly is not None:
        cols = [input_col, 'SVC TYPE', 'DETAIL REASON']
        try:
            df_oz_final = df_monthly[cols].copy()
            run_oz = True
        except: pass

    # Clean RAW DFs from RAM
    del df_daily, df_monthly
    gc.collect()

    # --- EXECUTE TRAINING ---
    
    # 1. DEFECT MODEL
    if run_defect:
        report(15, "Prep Defect Data...")
        df_ready = process_dataframe(df_defect_final)
        del df_defect_final; gc.collect()
        
        if len(df_ready) > 0:
            # Pass callback ke Embedder agar progress bar jalan saat encoding (20% -> 50%)
            embedder = BertEmbedder(MODEL_NAME, progress_callback=progress_callback, start_pct=20, end_pct=50)
            X_encoded = embedder.transform(df_ready['text_ready'].tolist())
            
            report(55, "Training RF (Defect)...")
            clf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
            clf.fit(X_encoded, df_ready[['Defect1', 'Defect2', 'Defect3']])
            
            # Save Pipeline (Simpan embedder polos tanpa callback agar bersih)
            clean_embedder = BertEmbedder(MODEL_NAME)
            final_pipe = Pipeline([('embedder', clean_embedder), ('clf', clf)])
            
            report(60, "Saving Defect Model...")
            joblib.dump(final_pipe, f'{MODEL_DIR}model_defect.pkl')
            del df_ready, X_encoded, clf, final_pipe; gc.collect()

    # 2. OZ MODEL
    if run_oz:
        report(65, "Prep Category Data...")
        df_ready = process_dataframe(df_oz_final)
        del df_oz_final; gc.collect()
        
        if len(df_ready) > 0:
            # Pass callback ke Embedder (70% -> 90%)
            embedder = BertEmbedder(MODEL_NAME, progress_callback=progress_callback, start_pct=70, end_pct=90)
            X_encoded = embedder.transform(df_ready['text_ready'].tolist())
            
            report(92, "Training RF (Category)...")
            clf = MultiOutputClassifier(RandomForestClassifier(**rf_params))
            clf.fit(X_encoded, df_ready[['SVC TYPE', 'DETAIL REASON']])
            
            clean_embedder = BertEmbedder(MODEL_NAME)
            final_pipe = Pipeline([('embedder', clean_embedder), ('clf', clf)])
            
            report(95, "Saving Category Model...")
            joblib.dump(final_pipe, f'{MODEL_DIR}model_oz.pkl')
            del df_ready, X_encoded, clf, final_pipe; gc.collect()

    report(100, "Training Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default False (Sesuai Request)
    parser.add_argument('--clean', action='store_true', default=False) 
    args = parser.parse_args()
    train_ai_advanced(enable_cleansing=args.clean)