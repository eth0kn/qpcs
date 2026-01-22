import pandas as pd
import joblib
import os
import numpy as np
import gc
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# CONFIGURATION & OPTIMIZATION (8 CPU / 16GB RAM)
# ==============================================================================
DATASET_PATH = 'datasets/training_data.xlsx'
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = 'BAAI/bge-m3' 

# CONFIG SHEET NAMES
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"

# Kunci untuk Auto Detect Header
HEADER_KEYWORD = "PROC_DETAIL_E"

# KOLOM PENTING UNTUK AI (CONTEXT FUSION)
# AI akan membaca gabungan kolom ini untuk akurasi maksimal
AI_CONTEXT_COLS = ['SYMPTOM_DESCRIPTION_E', 'PROC_DETAIL_E', 'ASC_REMARK_E']

# URUTAN KOLOM OUTPUT (Tampilan User)
FINAL_COLUMNS_ORDER = [
    'DATA_TYPE', 'RCPT_NO_ORD_NO', 'CLOSE_DT_RTN_DT', 'SALES_MODEL_SUFFIX', 
    'SERIAL_NO', 'PARTS_DESC1', 'PARTS_DESC2', 'PARTS_DESC3', 
    'PROC_DETAIL_E', 'ASC_REMARK_E'
]

# TARGET PREDIKSI
TARGETS_DEFECT = ['Defect1', 'Defect2', 'Defect3']
TARGETS_CATEGORY = ['SVC TYPE', 'DETAIL REASON']

# --- OPTIMASI CPU & RAM ---
torch.set_num_threads(8) 
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
AI_BATCH_SIZE = 256 

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def report(progress, message):
    print(f"[PROGRESS {progress}%] {message}")

def find_header_index(file_path, sheet_name, keyword):
    target_sheet = sheet_name if sheet_name is not None else 0
    try:
        df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=30)
    except:
        try:
            df_temp = pd.read_excel(file_path, sheet_name=0, header=None, nrows=30)
        except Exception as e:
            print(f"   ! Gagal baca header: {e}")
            return 1

    if isinstance(df_temp, dict): df_temp = list(df_temp.values())[0]

    for idx, row in df_temp.iterrows():
        row_str = row.astype(str).str.upper().tolist()
        if keyword.upper() in row_str:
            print(f"   > Header '{keyword}' ditemukan di baris ke-{idx + 1}")
            return idx
    return 1

def prepare_ai_input(df):
    """
    Menggabungkan kolom Konteks menjadi satu kalimat panjang untuk AI.
    Format: [SYMPTOM] [PROC_DETAIL] [REMARK]
    """
    # Pastikan kolom ada, jika tidak, isi string kosong
    for col in AI_CONTEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    
    # Gabungkan kolom (Fillna agar tidak error jika ada yg kosong)
    df['AI_INPUT_COMBINED'] = (
        df['SYMPTOM_DESCRIPTION_E'].fillna("").astype(str) + " " + 
        df['PROC_DETAIL_E'].fillna("").astype(str) + " " + 
        df['ASC_REMARK_E'].fillna("").astype(str)
    ).str.strip()
    
    return df

def load_and_validate_data(file_path, sheet_name):
    """Load data untuk TRAINING (Hanya ambil yg valid untuk belajar)"""
    if not os.path.exists(file_path): return pd.DataFrame()

    header_idx = find_header_index(file_path, sheet_name, HEADER_KEYWORD)
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    except:
        df = pd.read_excel(file_path, header=header_idx)

    if HEADER_KEYWORD not in df.columns: return pd.DataFrame()

    # 1. Siapkan Input Gabungan
    df = prepare_ai_input(df)

    # 2. Filter Sampah UNTUK TRAINING (Training butuh data bersih)
    # Kita filter berdasarkan 'AI_INPUT_COMBINED'
    invalid_values = ["", "0", "nan", "null", "-", "0.0"]
    mask_valid = ~df['AI_INPUT_COMBINED'].str.lower().isin(invalid_values)
    mask_len = df['AI_INPUT_COMBINED'].str.len() > 3 # Minimal 3 huruf
    
    initial_len = len(df)
    df = df[mask_valid & mask_len] # Training: Drop sampah
    final_len = len(df)
    
    if initial_len > final_len:
        print(f"   > Training Data: Dibuang {initial_len - final_len} baris tidak informatif.")
    
    return df

# ==============================================================================
# AI EMBEDDER (PERFORMANCE MODE)
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None): return self

    def transform(self, X):
        if self.model is None:
            print(f"   ...Loading Model BERT (8 Cores)...")
            self.model = SentenceTransformer(self.model_name, device='cpu')
        
        X_list = [str(text) if text is not None else "" for text in X]
        print(f"   ...Encoding {len(X_list)} kalimat (Batch Size: {AI_BATCH_SIZE})...")
        return self.model.encode(X_list, batch_size=AI_BATCH_SIZE, show_progress_bar=True)

# ==============================================================================
# ADVANCED STYLING ENGINE (NO WRAP)
# ==============================================================================
def apply_custom_layout(writer, df, sheet_name, report_type):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # STYLE DEFINITIONS
    title_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#4472C4', 'font_color': 'white',
        'align': 'center', 'valign': 'vcenter', 'font_size': 14, 'border': 1
    })
    header_std_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#F2F2F2', 'border': 1, 
        'align': 'center', 'valign': 'vcenter', 'text_wrap': False
    })
    header_defect_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#FFFF00', 'border': 1, 
        'align': 'center', 'valign': 'vcenter'
    })
    header_cat_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#FFC000', 'border': 1, 
        'align': 'center', 'valign': 'vcenter'
    })
    body_fmt = workbook.add_format({
        'border': 1, 'valign': 'top', 'text_wrap': False, 'align': 'left'
    })
    center_fmt = workbook.add_format({
        'border': 1, 'valign': 'top', 'align': 'center', 'text_wrap': False
    })

    fmt_ih = workbook.add_format({'bg_color': '#FF0000', 'font_color': 'white'})
    fmt_oz = workbook.add_format({'bg_color': '#00B050', 'font_color': 'white'})
    fmt_ms = workbook.add_format({'bg_color': '#FFFF00', 'font_color': 'black'}) 

    # POSISI & TITLE
    START_ROW = 1; START_COL = 1 
    title_text = "DEFECT CLASSIFICATION (DAILY 1X PER DAY)" if report_type == 'daily' else "OZ/MS/IH CATEGORY (MONTHLY 1X PER MONTH)"
    last_col_idx = START_COL + len(df.columns) - 1
    worksheet.merge_range(START_ROW, START_COL, START_ROW, last_col_idx, title_text, title_fmt)
    
    # HEADER
    header_row_idx = START_ROW + 1
    for i, col_name in enumerate(df.columns):
        col_idx = START_COL + i
        c_name = str(col_name).upper()
        
        if "DEFECT" in c_name: style = header_defect_fmt
        elif "SVC TYPE" in c_name or "REASON" in c_name: style = header_cat_fmt
        else: style = header_std_fmt
            
        worksheet.write(header_row_idx, col_idx, col_name, style)
        
        if "DESC" in c_name or "DETAIL" in c_name or "REMARK" in c_name:
            worksheet.set_column(col_idx, col_idx, 50)
        elif "NO" in c_name or "DATE" in c_name:
             worksheet.set_column(col_idx, col_idx, 15)
        else:
             worksheet.set_column(col_idx, col_idx, 20)

    # BODY
    data_start_row = header_row_idx + 1
    df_clean = df.fillna("")
    data_values = df_clean.values.tolist()
    
    for r_idx, row_data in enumerate(data_values):
        current_row = data_start_row + r_idx
        for c_idx, cell_value in enumerate(row_data):
            current_col = START_COL + c_idx
            col_head = df.columns[c_idx]
            if "DATA_TYPE" in col_head or "NO" in col_head:
                worksheet.write(current_row, current_col, cell_value, center_fmt)
            else:
                worksheet.write(current_row, current_col, cell_value, body_fmt)

    # CONDITIONAL FORMATTING
    if report_type == 'monthly' and 'SVC TYPE' in df.columns:
        svc_col_idx = df.columns.get_loc('SVC TYPE') + START_COL
        last_row = data_start_row + len(df) - 1
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx, {'type': 'cell', 'criteria': 'equal to', 'value': '"IH"', 'format': fmt_ih})
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx, {'type': 'cell', 'criteria': 'equal to', 'value': '"OZ"', 'format': fmt_oz})
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx, {'type': 'cell', 'criteria': 'equal to', 'value': '"MS"', 'format': fmt_ms})

    worksheet.freeze_panes(data_start_row, 0) 

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    global report
    if progress_callback:
        def report(p, m): progress_callback(p, m); print(f"[{p}%] {m}")

    report(10, "Menganalisa Struktur File & Load Data...")
    df_defect = load_and_validate_data(DATASET_PATH, SHEET_DAILY)
    df_oz = load_and_validate_data(DATASET_PATH, SHEET_MONTHLY)

    if not df_defect.empty:
        report(20, f"Training Model Defect ({len(df_defect)} rows)...")
        # Menggunakan kolom AI_INPUT_COMBINED
        pipeline = Pipeline([('embedder', BertEmbedder(MODEL_NAME)), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))])
        pipeline.fit(df_defect['AI_INPUT_COMBINED'].tolist(), df_defect[TARGETS_DEFECT].fillna("-"))
        joblib.dump(pipeline, f'{MODEL_DIR}model_defect.pkl')
    
    if not df_oz.empty:
        report(60, f"Training Model Category ({len(df_oz)} rows)...")
        pipeline_oz = Pipeline([('embedder', BertEmbedder(MODEL_NAME)), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))])
        pipeline_oz.fit(df_oz['AI_INPUT_COMBINED'].tolist(), df_oz[TARGETS_CATEGORY].fillna("-"))
        joblib.dump(pipeline_oz, f'{MODEL_DIR}model_oz.pkl')

    report(100, "Training Selesai.")

def predict_excel_process(input_file_path, report_type='daily'):
    import datetime
    
    # 1. READ
    try:
        header_idx = find_header_index(input_file_path, None, HEADER_KEYWORD)
        df = pd.read_excel(input_file_path, header=header_idx)
    except Exception as e:
        return None, f"Gagal baca file: {str(e)}"

    if HEADER_KEYWORD not in df.columns:
        return None, f"Kolom '{HEADER_KEYWORD}' tidak ditemukan."

    # 2. SIAPKAN KOLOM KONTEKS & FILTER LOGIC
    # Gabungkan kolom untuk otak AI
    df = prepare_ai_input(df)
    
    # Identifikasi Baris Valid vs Sampah
    invalid_values = ["", "0", "nan", "null", "-", "0.0"]
    mask_invalid = df['AI_INPUT_COMBINED'].str.lower().isin(invalid_values) | (df['AI_INPUT_COMBINED'].str.len() < 2)
    
    # Pisahkan Data: Valid untuk AI, Invalid untuk bypass
    # PENTING: Kita tidak drop row, hanya memisahkan index
    df_valid = df[~mask_invalid].copy()
    
    # 3. PASTIKAN KOLOM OUTPUT ADA
    for col in FINAL_COLUMNS_ORDER:
        if col not in df.columns: df[col] = "" 
    
    # Init kolom target dengan strip "-" (Default value)
    targets_all = TARGETS_DEFECT + (TARGETS_CATEGORY if report_type == 'monthly' else [])
    for t in targets_all:
        df[t] = "-"

    # 4. PREDICT DEFECT (Hanya untuk baris Valid)
    if not df_valid.empty:
        X_pred = df_valid['AI_INPUT_COMBINED'].tolist()
        
        try:
            model_def_path = f'{MODEL_DIR}model_defect.pkl'
            if os.path.exists(model_def_path):
                pipeline_def = joblib.load(model_def_path)
                y_pred_def = pipeline_def.predict(X_pred)
                
                # Update DataFrame Utama pada index yang valid saja
                df_res_def = pd.DataFrame(y_pred_def, columns=TARGETS_DEFECT, index=df_valid.index)
                df.update(df_res_def) # Update kolom Defect1-3 dengan hasil AI
        except Exception as e: return None, f"Error Model Defect: {e}"

        # 5. PREDICT CATEGORY (Hanya Monthly & Valid)
        if report_type == 'monthly':
            try:
                model_cat_path = f'{MODEL_DIR}model_oz.pkl'
                if os.path.exists(model_cat_path):
                    pipeline_cat = joblib.load(model_cat_path)
                    y_pred_cat = pipeline_cat.predict(X_pred)
                    
                    df_res_cat = pd.DataFrame(y_pred_cat, columns=TARGETS_CATEGORY, index=df_valid.index)
                    df.update(df_res_cat)
            except Exception as e: return None, f"Error Model Category: {e}"

    # 6. EXPORT (Buang kolom bantuan AI_INPUT_COMBINED, Sisakan format asli)
    # Gunakan FINAL_COLUMNS_ORDER + Kolom Target
    final_cols = FINAL_COLUMNS_ORDER + TARGETS_DEFECT
    if report_type == 'monthly':
        final_cols += TARGETS_CATEGORY
        
    # Pastikan kolom output ada di df (jika belum ada, buat kosong)
    for c in final_cols:
        if c not in df.columns: df[c] = ""
        
    df_final = df[final_cols].copy()

    # 7. SAVE EXCEL
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"RESULT_{report_type.upper()}_{timestamp}.xlsx"
    output_path = os.path.join("datasets", output_filename)
    
    try:
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        sheet_name = 'Defect Classification' if report_type == 'daily' else 'OZ,MS,IH CATEGORY'
        
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet
        
        apply_custom_layout(writer, df_final, sheet_name, report_type)
        writer.close()
    except Exception as e:
        return None, f"Gagal styling excel: {str(e)}"

    return output_path, "Success"