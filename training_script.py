import os
import pandas as pd
import numpy as np
import torch
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# CPU & THREAD CONFIG (OPTIMAL FOR 8 vCPU)
# ==============================================================================
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

AI_BATCH_SIZE = 256

# ==============================================================================
# LOAD SENTENCE-TRANSFORMER (OFFICIAL & STABLE)
# ==============================================================================
print("🔄 Loading SentenceTransformer model: BAAI/bge-m3")

EMBEDDER = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu"
)

print("✅ SentenceTransformer loaded")

# ==============================================================================
# CONSTANTS (ORIGINAL – TIDAK DIUBAH)
# ==============================================================================
DATASET_PATH = "datasets/training_data.xlsx"
MODEL_DIR = "models/"

SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"
HEADER_KEYWORD = "PROC_DETAIL_E"

AI_CONTEXT_COLS = [
    "SYMPTOM_DESCRIPTION_E",
    "PROC_DETAIL_E",
    "ASC_REMARK_E"
]

FINAL_COLUMNS_ORDER = [
    "DATA_TYPE", "RCPT_NO_ORD_NO", "CLOSE_DT_RTN_DT", "SALES_MODEL_SUFFIX",
    "SERIAL_NO", "PARTS_DESC1", "PARTS_DESC2", "PARTS_DESC3",
    "PROC_DETAIL_E", "ASC_REMARK_E"
]

TARGETS_DEFECT = ["Defect1", "Defect2", "Defect3"]
TARGETS_CATEGORY = ["SVC TYPE", "DETAIL REASON"]

# ==============================================================================
# HELPER FUNCTIONS (ORIGINAL LOGIC)
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

    df["AI_INPUT_COMBINED"] = (
        df["SYMPTOM_DESCRIPTION_E"].fillna("") + " " +
        df["PROC_DETAIL_E"].fillna("") + " " +
        df["ASC_REMARK_E"].fillna("")
    ).str.strip()

    return df

# ==============================================================================
# CUSTOM EMBEDDER (SINGLETON, NO RELOAD)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return EMBEDDER.encode(
            list(X),
            batch_size=AI_BATCH_SIZE,
            show_progress_bar=True
        )

# ==============================================================================
# TRAINING FUNCTION (ORIGINAL FLOW, STABILIZED)
# ==============================================================================
def train_ai_advanced(enable_cleansing=False, progress_callback=None):

    def report(progress, message):
        if progress_callback:
            progress_callback(progress, message)
        print(f"[{progress}%] {message}")

    # --------------------------------------------------------------------------
    report(10, "Loading defect training data...")
    df_defect = pd.read_excel(DATASET_PATH, sheet_name=SHEET_DAILY)
    df_defect = prepare_ai_input(df_defect)

    report(30, f"Training DEFECT model ({len(df_defect)} rows)...")

    pipe_defect = Pipeline([
        ("embedder", BertEmbedder()),
        ("clf", MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                n_jobs=8
            )
        ))
    ])

    pipe_defect.fit(
        df_defect["AI_INPUT_COMBINED"],
        df_defect[TARGETS_DEFECT].fillna("-")
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe_defect, os.path.join(MODEL_DIR, "model_defect.pkl"))

    # --------------------------------------------------------------------------
    report(60, "Loading category training data...")
    df_cat = pd.read_excel(DATASET_PATH, sheet_name=SHEET_MONTHLY)
    df_cat = prepare_ai_input(df_cat)

    report(80, f"Training CATEGORY model ({len(df_cat)} rows)...")

    pipe_cat = Pipeline([
        ("embedder", BertEmbedder()),
        ("clf", MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                n_jobs=8
            )
        ))
    ])

    pipe_cat.fit(
        df_cat["AI_INPUT_COMBINED"],
        df_cat[TARGETS_CATEGORY].fillna("-")
    )

    joblib.dump(pipe_cat, os.path.join(MODEL_DIR, "model_oz.pkl"))

    report(100, "✅ Training completed successfully")
