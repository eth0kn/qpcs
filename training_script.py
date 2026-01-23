import os
import gc
import threading
import pandas as pd
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "training_data.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"
AI_BATCH_SIZE = 128

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# MODEL SINGLETON (HYBRID OPTIMIZATION)
# =============================================================================
_model_lock = threading.Lock()
_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        with _model_lock:
            if _EMBEDDER is None:
                print("🔄 Loading SentenceTransformer:", MODEL_NAME)
                _EMBEDDER = SentenceTransformer(MODEL_NAME, device="cpu")
                print("✅ SentenceTransformer loaded")
    return _EMBEDDER


# =============================================================================
# CONSTANTS (ORIGINAL)
# =============================================================================
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"

AI_CONTEXT_COLS = [
    "SYMPTOM_DESCRIPTION_E",
    "PROC_DETAIL_E",
    "ASC_REMARK_E",
]

TARGETS_DEFECT = ["Defect1", "Defect2", "Defect3"]
TARGETS_CATEGORY = ["SVC TYPE", "DETAIL REASON"]

# =============================================================================
# UTIL FUNCTIONS (ORIGINAL – TIDAK DIHAPUS)
# =============================================================================
def find_header_index(file_path, sheet_name, keyword):
    df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=30)
    for idx, row in df_temp.iterrows():
        if keyword.upper() in row.astype(str).str.upper().tolist():
            return idx
    return 1


def prepare_ai_input(df):
    for col in AI_CONTEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    df["AI_INPUT_COMBINED"] = (
        df["SYMPTOM_DESCRIPTION_E"].fillna("") + " "
        + df["PROC_DETAIL_E"].fillna("") + " "
        + df["ASC_REMARK_E"].fillna("")
    ).str.strip()
    return df


# =============================================================================
# EMBEDDER (PIPELINE SAFE)
# =============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embedder = get_embedder()
        return embedder.encode(
            list(X),
            batch_size=AI_BATCH_SIZE,
            show_progress_bar=True
        )


# =============================================================================
# TRAINING (ORIGINAL LOGIC + SAFE GC)
# =============================================================================
def train_ai_advanced(enable_cleansing=False, progress_callback=None):

    def report(p, m):
        if progress_callback:
            progress_callback(p, m)
        print(f"[{p}%] {m}")

    report(10, "Loading DEFECT dataset...")
    df_defect = pd.read_excel(DATASET_PATH, sheet_name=SHEET_DAILY)
    df_defect = prepare_ai_input(df_defect)

    report(30, "Training DEFECT model...")
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
    joblib.dump(pipe_defect, os.path.join(MODEL_DIR, "model_defect.pkl"))

    # cleanup
    del df_defect
    del pipe_defect
    gc.collect()

    report(60, "Loading CATEGORY dataset...")
    df_cat = pd.read_excel(DATASET_PATH, sheet_name=SHEET_MONTHLY)
    df_cat = prepare_ai_input(df_cat)

    report(80, "Training CATEGORY model...")
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

    del df_cat
    del pipe_cat
    gc.collect()

    report(100, "✅ Training completed")


# =============================================================================
# PREDICTION (ORIGINAL – DIPERTAHANKAN)
# =============================================================================
def predict_excel_process(excel_path: str) -> dict:
    model_defect = joblib.load(os.path.join(MODEL_DIR, "model_defect.pkl"))
    model_cat = joblib.load(os.path.join(MODEL_DIR, "model_oz.pkl"))

    df = pd.read_excel(excel_path)
    df = prepare_ai_input(df)

    pred_defect = model_defect.predict(df["AI_INPUT_COMBINED"])
    pred_cat = model_cat.predict(df["AI_INPUT_COMBINED"])

    del model_defect
    del model_cat
    gc.collect()

    return {
        "defect": pred_defect.tolist(),
        "category": pred_cat.tolist()
    }


# =============================================================================
# EXCEL STYLING (ORIGINAL FEATURE – TIDAK HILANG)
# =============================================================================
def apply_custom_layout(file_path: str):
    wb = load_workbook(file_path)
    ws = wb.active

    header_fill = PatternFill("solid", fgColor="BDD7EE")
    header_font = Font(bold=True)
    center_align = Alignment(vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = thin_border

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = center_align
            cell.border = thin_border

    for col in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 40)

    ws.freeze_panes = "A2"
    wb.save(file_path)
