import pandas as pd
import joblib
import os
import numpy as np
import re
import argparse # PENTING: Untuk menangkap argumen --no-clean dari terminal
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

# ==============================================================================
# AI EMBEDDER
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
# MAIN TRAINING LOGIC
# ==============================================================================
def train_ai_advanced(enable_cleansing=True, progress_callback=None):
    """
    Core function called by Backend OR Terminal.
    """
    def report(p, msg):
        if progress_callback: progress_callback(p, msg)
        print(f"[{p}%] {msg}")

    mode_msg = "ON (Deep Clean)" if enable_cleansing else "OFF (Raw Data)"
    report(5, f"Initializing Training (Cleansing: {mode_msg})...")

    if not os.path.exists(DATASET_PATH):
        report(0, "Error: Dataset not found.")
        return

    # --- STEP 1: LOAD ---
    report(10, "Reading Excel Sheets (Daily & Monthly)...")
    try:
        df_oz = pd.read_excel(DATASET_PATH, sheet_name='OZ,MS,IH CATEGORY', header=1, engine='openpyxl')
        df_daily = pd.read_excel(DATASET_PATH, sheet_name='Defect Classification', header=2, engine='openpyxl')
    except Exception as e:
        report(0, f"Error reading Excel: {str(e)}")
        return

    # --- STEP 2: PREPARE ---
    report(25, "Merging & Organizing Data...")
    input_col = 'PROC_DETAIL_E'
    
    cols_defect = [input_col, 'Defect1', 'Defect2', 'Defect3']
    df_defect_train = pd.concat([df_oz[cols_defect], df_daily[cols_defect]], ignore_index=True)
    
    cols_cat = [input_col, 'SVC TYPE', 'DETAIL REASON']
    df_cat_train = df_oz[cols_cat].copy()

    # --- STEP 3: CLEANSING (DYNAMIC) ---
    report(40, f"Processing Text (Cleansing={enable_cleansing})...")

    def process_dataframe(df):
        df[input_col] = df[input_col].fillna("").astype(str)
        
        # LOGIKA PERCABANGAN DI SINI
        if enable_cleansing:
            # Jika ON: Jalankan Regex & Filter Sampah
            df['text_ready'] = df[input_col].apply(clean_text_deep)
            df['valid'] = df['text_ready'].apply(is_valid_training_row)
        else:
            # Jika OFF: Pakai Data Mentah & Anggap Semua Valid
            df['text_ready'] = df[input_col]
            df['valid'] = True 
            
        return df[df['valid'] == True].copy().fillna('unknown')

    df_defect_ready = process_dataframe(df_defect_train)
    df_cat_ready = process_dataframe(df_cat_train)

    if len(df_defect_ready) == 0:
        report(0, "Error: No valid data available.")
        return

    # --- STEP 4: TRAINING ---
    rf_params = {'n_estimators': 200, 'n_jobs': -1, 'random_state': 42}
    
    report(60, "Training Neural Brain (Defect Model)...")
    pipe_defect = Pipeline([
        ('embedder', BertEmbedder(MODEL_NAME)), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(**rf_params)))
    ])
    pipe_defect.fit(df_defect_ready['text_ready'], df_defect_ready[['Defect1', 'Defect2', 'Defect3']])

    report(80, "Training Neural Brain (Category Model)...")
    pipe_oz = Pipeline([
        ('embedder', BertEmbedder(MODEL_NAME)), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(**rf_params)))
    ])
    pipe_oz.fit(df_cat_ready['text_ready'], df_cat_ready[['SVC TYPE', 'DETAIL REASON']])

    # --- STEP 5: SAVE ---
    report(95, "Saving Models to Disk...")
    joblib.dump(pipe_defect, f'{MODEL_DIR}model_defect.pkl')
    joblib.dump(pipe_oz, f'{MODEL_DIR}model_oz.pkl')

    report(100, "Training Complete! Models Updated.")

# ==============================================================================
# CLI HANDLER (Supaya bisa dijalankan manual lewat terminal)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Models Manually")
    
    # Menambahkan opsi --no-clean
    parser.add_argument('--clean', action='store_true', help='Enable Deep Cleansing (Default)')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Disable Cleansing (Use Raw Data)')
    parser.set_defaults(clean=True) # Defaultnya adalah TRUE (Membersihkan data)
    
    args = parser.parse_args()
    
    # Jalankan fungsi utama
    train_ai_advanced(enable_cleansing=args.clean)