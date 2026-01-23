import pandas as pd
import joblib
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# PATH & CPU CONFIG (NEW)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models_export", "bge-m3")

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
AI_BATCH_SIZE = 256

# ==============================================================================
# PRELOAD MODEL (NEW – OFFLINE & SINGLETON)
# ==============================================================================
print(f"🔄 Loading SentenceTransformer snapshot: {SNAPSHOT}")
EMBEDDER = SentenceTransformer(
    MODEL_PATH,
    device="cpu",
    local_files_only=True
)
print("✅ SentenceTransformer loaded")

# ==============================================================================
# ORIGINAL CONSTANTS (UNCHANGED)
# ==============================================================================
DATASET_PATH = 'datasets/training_data.xlsx'
MODEL_DIR = 'models/'
MODEL_NAME = 'BAAI/bge-m3'

SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"
HEADER_KEYWORD = "PROC_DETAIL_E"

AI_CONTEXT_COLS = ['SYMPTOM_DESCRIPTION_E', 'PROC_DETAIL_E', 'ASC_REMARK_E']
FINAL_COLUMNS_ORDER = [
    'DATA_TYPE', 'RCPT_NO_ORD_NO', 'CLOSE_DT_RTN_DT', 'SALES_MODEL_SUFFIX',
    'SERIAL_NO', 'PARTS_DESC1', 'PARTS_DESC2', 'PARTS_DESC3',
    'PROC_DETAIL_E', 'ASC_REMARK_E'
]

TARGETS_DEFECT = ['Defect1', 'Defect2', 'Defect3']
TARGETS_CATEGORY = ['SVC TYPE', 'DETAIL REASON']

# ==============================================================================
# ORIGINAL FUNCTIONS (UNCHANGED)
# ==============================================================================
def find_header_index(file_path, sheet_name, keyword):
    df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=30)
    if isinstance(df_temp, dict):
        df_temp = list(df_temp.values())[0]
    for idx, row in df_temp.iterrows():
        if keyword.upper() in row.astype(str).str.upper().tolist():
            return idx
    return 1

def prepare_ai_input(df):
    for col in AI_CONTEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    df['AI_INPUT_COMBINED'] = (
        df['SYMPTOM_DESCRIPTION_E'].fillna("") + " " +
        df['PROC_DETAIL_E'].fillna("") + " " +
        df['ASC_REMARK_E'].fillna("")
    ).str.strip()
    return df

# ==============================================================================
# PATCHED EMBEDDER (ONLY CHANGE INSIDE PIPELINE)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return EMBEDDER.encode(
            list(X),
            batch_size=AI_BATCH_SIZE,
            show_progress_bar=True
        )

# ==============================================================================
# TRAIN (LOGIC UNCHANGED)
# ==============================================================================
def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    def report(p, m):
        if progress_callback:
            progress_callback(p, m)
        print(f"[{p}%] {m}")

    report(10, "Analyzing File & Load Data...")
    df_defect = pd.read_excel(DATASET_PATH, sheet_name=SHEET_DAILY)
    df_defect = prepare_ai_input(df_defect)

    report(20, f"Training Model Defect ({len(df_defect)})...")
    pipe_defect = Pipeline([
        ('embedder', BertEmbedder()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                n_jobs=8
            )
        ))
    ])
    pipe_defect.fit(df_defect['AI_INPUT_COMBINED'], df_defect[TARGETS_DEFECT].fillna("-"))
    joblib.dump(pipe_defect, f'{MODEL_DIR}model_defect.pkl')

    df_cat = pd.read_excel(DATASET_PATH, sheet_name=SHEET_MONTHLY)
    df_cat = prepare_ai_input(df_cat)

    report(60, f"Training Model Category ({len(df_cat)})...")
    pipe_cat = Pipeline([
        ('embedder', BertEmbedder()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                n_jobs=8
            )
        ))
    ])
    pipe_cat.fit(df_cat['AI_INPUT_COMBINED'], df_cat[TARGETS_CATEGORY].fillna("-"))
    joblib.dump(pipe_cat, f'{MODEL_DIR}model_oz.pkl')

    report(100, "Training Completed")
