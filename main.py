from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
import os
import shutil
import re
import threading
import time

# --- LIBRARIES ---
from openpyxl.styles import PatternFill, Border, Side, Alignment, Font
from openpyxl.utils import get_column_letter
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# AI EMBEDDER CLASS (Untuk Load Model saat Prediksi)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    # Constructor disesuaikan agar kompatibel dengan training_script
    def __init__(self, model_name, progress_callback=None, start_pct=0, end_pct=0):
        self.model_name = model_name
        self.model = None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.model is None: self.model = SentenceTransformer(self.model_name)
        if hasattr(X, 'tolist'): sentences = X.tolist()
        else: sentences = X
        # Batch size 64
        return self.model.encode(sentences, batch_size=64, show_progress_bar=False)

app = FastAPI(title="QPCS AI System API", version="18.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = 'models/'
DATASET_DIR = 'datasets/'
MODEL_DEFECT_PATH = os.path.join(MODEL_DIR, 'model_defect.pkl')
MODEL_OZ_PATH = os.path.join(MODEL_DIR, 'model_oz.pkl')

ai_models = {}
TRAINING_STATE = {"is_running": False, "progress": 0, "message": "Idle"}

# ... (Helper functions load_system_resources, clean_text_deep, is_valid_text sama) ...
def load_system_resources():
    print("🚀 [SYSTEM START] Loading AI Models...")
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
    if not s or s.lower() in ["nan", "null"]: return False
    return True

# ... (Helper load_dataset_auto_header, filter_output_columns, apply_excel_styling SAMA SEPERTI SEBELUMNYA) ...
def load_dataset_auto_header(file_contents):
    try:
        temp_df = pd.read_excel(io.BytesIO(file_contents), header=None, nrows=10, engine='openpyxl')
        header_idx = 0
        keywords = ['DATA_TYPE', 'PROC_DETAIL_E', 'SERIAL_NO', 'PARTS_DESC1', 'CLOSE_DT_RTN_DT']
        for idx, row in temp_df.iterrows():
            row_str = [str(val).strip().upper() for val in row.values]
            if sum(1 for k in keywords if k in row_str) >= 2:
                header_idx = idx
                break
        return pd.read_excel(io.BytesIO(file_contents), header=header_idx, engine='openpyxl')
    except: return pd.read_excel(io.BytesIO(file_contents), header=0, engine='openpyxl')

def filter_output_columns(df, input_col_name):
    required_cols = ['DATA_TYPE', 'RCPT_NO_ORD_NO', 'CLOSE_DT_RTN_DT', 'SALES_MODEL_SUFFIX', 'SERIAL_NO', 'PARTS_DESC1', 'PARTS_DESC2', 'PARTS_DESC3', 'PROC_DETAIL_E', 'ASC_REMARK_E']
    df_clean = pd.DataFrame()
    for col in required_cols:
        df_clean[col] = df[col] if col in df.columns else ""
    if 'PROC_DETAIL_E' not in df_clean.columns and input_col_name in df.columns:
        df_clean['PROC_DETAIL_E'] = df[input_col_name]
    return df_clean

def apply_excel_styling(ws, report_type, header_row_num, start_col_idx):
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    header_fill = PatternFill(start_color="4F46E5", end_color="4F46E5", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    fill_oz = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    fill_ih = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    fill_ms = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    svc_col_index = None

    for row in ws.iter_rows(min_row=header_row_num, max_row=header_row_num, min_col=start_col_idx):
        for cell in row:
            if cell.value:
                cell.font = header_font; cell.fill = header_fill; cell.alignment = Alignment(horizontal="center", vertical="center"); cell.border = thin_border
                if report_type == 'monthly' and str(cell.value).strip() == 'SVC TYPE': svc_col_index = cell.column

    for row in ws.iter_rows(min_row=header_row_num + 1, min_col=start_col_idx):
        for cell in row:
            if ws.cell(row=header_row_num, column=cell.column).value: 
                cell.border = thin_border; cell.alignment = Alignment(vertical="center")
                if report_type == 'monthly' and svc_col_index and cell.column == svc_col_index:
                    val = str(cell.value).strip().upper()
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    if val == 'OZ': cell.fill = fill_oz; cell.font = Font(color="006100")
                    elif val == 'IH': cell.fill = fill_ih; cell.font = Font(color="9C0006")
                    elif val == 'MS': cell.fill = fill_ms; cell.font = Font(color="9C6500")

    for column in ws.columns:
        if column[0].column < start_col_idx: continue
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            try:
                if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
            except: pass
        adjusted_width = (max_length + 2) if (max_length + 2) < 50 else 50
        ws.column_dimensions[column_letter].width = adjusted_width

# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.post("/predict")
async def predict_excel(
    file: UploadFile = File(...),
    report_type: str = Query(..., description="'daily' or 'monthly'"),
    enable_cleansing: bool = Query(False) # DEFAULT FALSE
):
    if not file.filename.endswith(('.xlsx', '.xls')): raise HTTPException(400, "Invalid file format.")
    try:
        contents = await file.read()
        df_raw = load_dataset_auto_header(contents)
        
        input_col = 'PROC_DETAIL_E'
        if input_col not in df_raw.columns:
             candidates = [c for c in df_raw.columns if 'detail' in str(c).lower()]
             if candidates: input_col = candidates[0]
        
        df_final = filter_output_columns(df_raw, input_col)
        
        X_temp = df_final['PROC_DETAIL_E'].fillna("").astype(str)
        if enable_cleansing: X_temp = X_temp.apply(clean_text_deep)
        
        valid_mask = X_temp.apply(is_valid_text)
        X_pred = X_temp[valid_mask].tolist()

        # PREDICT LOGIC (Daily/Monthly) ...
        # (LOGIC SAMA SEPERTI V15, HANYA MEMASTIKAN DEFAULT CLEANSING FALSE)
        if report_type == 'daily':
            df_final['Defect1'] = "-"; df_final['Defect2'] = "-"; df_final['Defect3'] = "-"
            if len(X_pred) > 0 and 'defect' in ai_models:
                 pred_def = ai_models['defect'].predict(X_pred)
                 df_final.loc[valid_mask, 'Defect1'] = pred_def[:, 0]
                 df_final.loc[valid_mask, 'Defect2'] = pred_def[:, 1]
                 df_final.loc[valid_mask, 'Defect3'] = pred_def[:, 2]
            sheet_name = 'Defect Classification'; start_row = 2; start_col = 1; header_cell = "B2"; header_title = "DEFECT CLASSIFICATION (DAILY 1X PER DAY)"
        
        elif report_type == 'monthly':
            df_final['Defect1'] = "-"; df_final['Defect2'] = "-"; df_final['Defect3'] = "-"; df_final['SVC TYPE'] = "-"; df_final['DETAIL REASON'] = "-"
            if len(X_pred) > 0:
                if 'defect' in ai_models:
                    pred_def = ai_models['defect'].predict(X_pred)
                    df_final.loc[valid_mask, 'Defect1'] = pred_def[:, 0]; df_final.loc[valid_mask, 'Defect2'] = pred_def[:, 1]; df_final.loc[valid_mask, 'Defect3'] = pred_def[:, 2]
                if 'oz' in ai_models:
                    pred_oz = ai_models['oz'].predict(X_pred)
                    df_final.loc[valid_mask, 'SVC TYPE'] = pred_oz[:, 0]; df_final.loc[valid_mask, 'DETAIL REASON'] = pred_oz[:, 1]
            sheet_name = 'OZ,MS,IH CATEGORY'; start_row = 1; start_col = 0; header_cell = "A1"; header_title = "OZ/MS/IH CATEGORY (MONTHLY 1X PER MONTH)"

        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=start_col)
            worksheet = writer.sheets[sheet_name]
            worksheet[header_cell] = header_title
            worksheet[header_cell].font = Font(bold=True, size=12)
            apply_excel_styling(worksheet, report_type, header_row_num=(start_row + 1), start_col_idx=(start_col + 1))
        
        output_buffer.seek(0)
        prefix = "DAILY_" if report_type == 'daily' else "MONTHLY_"
        filename = f"{prefix}RESULT_{file.filename}"
        return StreamingResponse(output_buffer, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={"Content-Disposition": f"attachment; filename={filename}"})

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, f"Server Error: {str(e)}")

# TRAINING ENDPOINT (DEFAULT CLEAN = FALSE)
def run_training_process(clean_bool):
    global TRAINING_STATE
    TRAINING_STATE["is_running"] = True; TRAINING_STATE["progress"] = 0; TRAINING_STATE["message"] = "Initializing..."
    try:
        from training_script import train_ai_advanced
        def update_progress(pct, msg):
            TRAINING_STATE["progress"] = pct; TRAINING_STATE["message"] = msg
        train_ai_advanced(enable_cleansing=clean_bool, progress_callback=update_progress)
        load_system_resources()
    except Exception as e:
        TRAINING_STATE["message"] = f"Error: {str(e)}"; TRAINING_STATE["progress"] = 0
    finally:
        TRAINING_STATE["is_running"] = False

@app.post("/train")
async def start_training(file: UploadFile = File(...), enable_cleansing: bool = Query(False)): # DEFAULT FALSE
    global TRAINING_STATE
    if TRAINING_STATE["is_running"]: raise HTTPException(400, "Training in progress.")
    os.makedirs(DATASET_DIR, exist_ok=True)
    try:
        with open(os.path.join(DATASET_DIR, "training_data.xlsx"), "wb+") as fo: shutil.copyfileobj(file.file, fo)
    except: raise HTTPException(400, "File locked.")
    thread = threading.Thread(target=run_training_process, args=(enable_cleansing,))
    thread.start()
    return {"status": "started"}

@app.get("/train/status")
async def get_training_status(): return TRAINING_STATE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)