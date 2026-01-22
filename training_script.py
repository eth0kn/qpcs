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
# CONFIGURATION & OPTIMIZATION
# ==============================================================================
DATASET_PATH = 'datasets/training_data.xlsx'
MODEL_DIR = 'models/'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = 'BAAI/bge-m3' 

# CONFIG SHEET NAMES (Sesuaikan dengan nama sheet di Raw Data User)
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"

# Kunci untuk Auto Detect Header
HEADER_KEYWORD = "PROC_DETAIL_E"

# URUTAN KOLOM OUTPUT YANG DIINGINKAN (Sesuai Expected Result)
FINAL_COLUMNS_ORDER = [
    'DATA_TYPE', 'RCPT_NO_ORD_NO', 'CLOSE_DT_RTN_DT', 'SALES_MODEL_SUFFIX', 
    'SERIAL_NO', 'PARTS_DESC1', 'PARTS_DESC2', 'PARTS_DESC3', 
    'PROC_DETAIL_E', 'ASC_REMARK_E'
]

# TARGET PREDIKSI
TARGETS_DEFECT = ['Defect1', 'Defect2', 'Defect3']
TARGETS_CATEGORY = ['SVC TYPE', 'DETAIL REASON']

# --- OPTIMASI CPU ---
torch.set_num_threads(8) 
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def report(progress, message):
    print(f"[PROGRESS {progress}%] {message}")

def find_header_index(file_path, sheet_name, keyword):
    """
    Mencari lokasi header (baris ke-berapa) yang mengandung keyword.
    FIX: Menangani kasus sheet_name=None agar tidak return Dict.
    """
    # FIX: Jika sheet_name None, gunakan index 0 (Sheet Pertama)
    target_sheet = sheet_name if sheet_name is not None else 0

    try:
        # Baca 30 baris pertama untuk scanning
        df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=30)
    except:
        try:
            # Fallback jika nama sheet salah, paksa index 0
            df_temp = pd.read_excel(file_path, sheet_name=0, header=None, nrows=30)
        except Exception as e:
            print(f"   ! Gagal baca header: {e}")
            return 1

    # Safety: Jika pandas mengembalikan dict (kasus edge case), ambil value pertama
    if isinstance(df_temp, dict):
        df_temp = list(df_temp.values())[0]

    for idx, row in df_temp.iterrows():
        row_str = row.astype(str).str.upper().tolist()
        if keyword.upper() in row_str:
            print(f"   > Header '{keyword}' ditemukan di baris ke-{idx + 1}")
            return idx
    return 1

def load_and_validate_data(file_path, sheet_name):
    """Load data untuk Training"""
    if not os.path.exists(file_path): return pd.DataFrame()

    header_idx = find_header_index(file_path, sheet_name, HEADER_KEYWORD)
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    except:
        # Fallback sheet pertama
        df = pd.read_excel(file_path, header=header_idx)

    if HEADER_KEYWORD not in df.columns: return pd.DataFrame()

    # Validasi Basic
    df[HEADER_KEYWORD] = df[HEADER_KEYWORD].fillna("").astype(str)
    df = df[df[HEADER_KEYWORD].str.strip().str.len() > 1]
    
    return df

# ==============================================================================
# AI EMBEDDER
# ==============================================================================
class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            print(f"   ...Loading Model BERT (8 Cores)...")
            self.model = SentenceTransformer(self.model_name, device='cpu')
        
        X = [str(text) if text is not None else "" for text in X]
        print(f"   ...Encoding {len(X)} kalimat...")
        return self.model.encode(X, batch_size=128, show_progress_bar=True)

# ==============================================================================
# ADVANCED STYLING ENGINE (XlsxWriter)
# ==============================================================================
def apply_custom_layout(writer, df, sheet_name, report_type):
    """
    Melakukan formatting presisi: Row 2, Col B, Merge Title, Colors.
    """
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # --- 1. DEFINISI FORMAT WARNA ---
    
    # Title Header (Biru #4472C4, Putih, Bold)
    title_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#4472C4', 'font_color': 'white',
        'align': 'center', 'valign': 'vcenter', 'font_size': 14, 'border': 1
    })

    # Header Standar (Abu-abu muda biar rapi)
    header_std_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#F2F2F2', 'border': 1, 
        'align': 'center', 'valign': 'vcenter', 'text_wrap': True
    })

    # Header Defect (Kuning #FFFF00)
    header_defect_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#FFFF00', 'border': 1, 
        'align': 'center', 'valign': 'vcenter'
    })

    # Header Category (Orange #FFC000) - Khusus Monthly
    header_cat_fmt = workbook.add_format({
        'bold': True, 'fg_color': '#FFC000', 'border': 1, 
        'align': 'center', 'valign': 'vcenter'
    })

    # Body Data
    body_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'text_wrap': True})
    center_fmt = workbook.add_format({'border': 1, 'valign': 'top', 'align': 'center'})

    # Conditional Formats (IH=Merah, OZ=Hijau, MS=Kuning)
    # Note: MS Kuning font hitam agar terbaca. IH/OZ Font Putih.
    fmt_ih = workbook.add_format({'bg_color': '#FF0000', 'font_color': 'white'})
    fmt_oz = workbook.add_format({'bg_color': '#00B050', 'font_color': 'white'})
    fmt_ms = workbook.add_format({'bg_color': '#FFFF00', 'font_color': 'black'}) 

    # --- 2. SETUP POSISI (Start Row 2, Col B) ---
    START_ROW = 1 # Baris ke-2 (Index 1)
    START_COL = 1 # Kolom B (Index 1)
    
    # --- 3. TULIS JUDUL UTAMA (MERGED) ---
    title_text = ""
    last_col_idx = START_COL + len(df.columns) - 1
    
    if report_type == 'daily':
        title_text = "DEFECT CLASSIFICATION (DAILY 1X PER DAY)"
    else:
        title_text = "OZ/MS/IH CATEGORY (MONTHLY 1X PER MONTH)"
    
    # Merge Range: (FirstRow, FirstCol, LastRow, LastCol)
    worksheet.merge_range(START_ROW, START_COL, START_ROW, last_col_idx, title_text, title_fmt)
    
    # --- 4. TULIS HEADER TABEL (ROW 3 / Index 2) ---
    header_row_idx = START_ROW + 1
    
    for i, col_name in enumerate(df.columns):
        col_idx = START_COL + i
        c_name = str(col_name).upper()
        
        # Tentukan Style Header
        if "DEFECT" in c_name:
            style = header_defect_fmt
        elif "SVC TYPE" in c_name or "REASON" in c_name:
            style = header_cat_fmt
        else:
            style = header_std_fmt
            
        worksheet.write(header_row_idx, col_idx, col_name, style)
        
        # Atur Lebar Kolom
        if "DESC" in c_name or "DETAIL" in c_name or "REMARK" in c_name:
            worksheet.set_column(col_idx, col_idx, 40) # Lebar untuk deskripsi
        elif "NO" in c_name or "DATE" in c_name:
             worksheet.set_column(col_idx, col_idx, 15)
        else:
             worksheet.set_column(col_idx, col_idx, 20)

    # --- 5. TULIS DATA BODY (ROW 4 dst) ---
    data_start_row = header_row_idx + 1
    
    # Ubah DF ke list of list & handle NaN agar aman ditulis
    df_clean = df.fillna("")
    data_values = df_clean.values.tolist()
    
    for r_idx, row_data in enumerate(data_values):
        current_row = data_start_row + r_idx
        for c_idx, cell_value in enumerate(row_data):
            current_col = START_COL + c_idx
            
            # Cek apakah kolom perlu Center Align
            col_head = df.columns[c_idx]
            if "DATA_TYPE" in col_head or "NO" in col_head:
                worksheet.write(current_row, current_col, cell_value, center_fmt)
            else:
                worksheet.write(current_row, current_col, cell_value, body_fmt)

    # --- 6. APPLY CONDITIONAL FORMATTING (Khusus Monthly) ---
    if report_type == 'monthly' and 'SVC TYPE' in df.columns:
        # Cari index kolom SVC TYPE
        svc_col_idx = df.columns.get_loc('SVC TYPE') + START_COL
        
        # Convert index ke huruf excel (misal 5 -> F)
        last_row = data_start_row + len(df) - 1
        
        # Apply Logic
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx,
                                     {'type': 'cell', 'criteria': 'equal to', 'value': '"IH"', 'format': fmt_ih})
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx,
                                     {'type': 'cell', 'criteria': 'equal to', 'value': '"OZ"', 'format': fmt_oz})
        worksheet.conditional_format(data_start_row, svc_col_idx, last_row, svc_col_idx,
                                     {'type': 'cell', 'criteria': 'equal to', 'value': '"MS"', 'format': fmt_ms})

    # Freeze Panes (Bekukan Header Table)
    worksheet.freeze_panes(data_start_row, 0) 

# ==============================================================================
# MAIN LOGIC (TRAIN & PREDICT)
# ==============================================================================

def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    global report
    if progress_callback:
        def report(p, m): progress_callback(p, m); print(f"[{p}%] {m}")

    report(10, "Menganalisa Struktur File & Load Data...")

    df_defect = load_and_validate_data(DATASET_PATH, SHEET_DAILY)
    df_oz = load_and_validate_data(DATASET_PATH, SHEET_MONTHLY)

    # --- TRAIN DEFECT ---
    if not df_defect.empty:
        report(20, f"Training Model Defect ({len(df_defect)} rows)...")
        pipeline = Pipeline([('embedder', BertEmbedder(MODEL_NAME)), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))])
        pipeline.fit(df_defect[HEADER_KEYWORD].tolist(), df_defect[TARGETS_DEFECT].fillna("-"))
        joblib.dump(pipeline, f'{MODEL_DIR}model_defect.pkl')
    
    # --- TRAIN CATEGORY ---
    if not df_oz.empty:
        report(60, f"Training Model Category ({len(df_oz)} rows)...")
        pipeline_oz = Pipeline([('embedder', BertEmbedder(MODEL_NAME)), ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))])
        pipeline_oz.fit(df_oz[HEADER_KEYWORD].tolist(), df_oz[TARGETS_CATEGORY].fillna("-"))
        joblib.dump(pipeline_oz, f'{MODEL_DIR}model_oz.pkl')

    report(100, "Training Selesai.")

def predict_excel_process(input_file_path, report_type='daily'):
    import datetime
    
    # 1. LOAD DATA & HEADER (FIX: Pass None as sheet_name handled in func)
    try:
        header_idx = find_header_index(input_file_path, None, HEADER_KEYWORD)
        df = pd.read_excel(input_file_path, header=header_idx)
    except Exception as e:
        return None, f"Gagal baca file: {str(e)}"

    if HEADER_KEYWORD not in df.columns:
        return None, f"Kolom '{HEADER_KEYWORD}' tidak ditemukan."

    # 2. FILTER & URUTKAN KOLOM
    for col in FINAL_COLUMNS_ORDER:
        if col not in df.columns:
            df[col] = "" 
    
    df_final = df[FINAL_COLUMNS_ORDER].copy()
    X_pred = df[HEADER_KEYWORD].fillna("").astype(str).tolist()

    # 3. RUN PREDICTION (DEFECT - Selalu jalan)
    try:
        model_def_path = f'{MODEL_DIR}model_defect.pkl'
        if not os.path.exists(model_def_path): return None, "Model Defect belum ada."
        
        pipeline_def = joblib.load(model_def_path)
        y_pred_def = pipeline_def.predict(X_pred)
        
        df_res_def = pd.DataFrame(y_pred_def, columns=TARGETS_DEFECT)
        for c in TARGETS_DEFECT: df_final[c] = df_res_def[c]
        
    except Exception as e: return None, f"Error Model Defect: {e}"

    # 4. RUN PREDICTION (CATEGORY - Hanya Monthly)
    if report_type == 'monthly':
        try:
            model_cat_path = f'{MODEL_DIR}model_oz.pkl'
            if not os.path.exists(model_cat_path): return None, "Model Category belum ada."
            
            pipeline_cat = joblib.load(model_cat_path)
            y_pred_cat = pipeline_cat.predict(X_pred)
            
            df_res_cat = pd.DataFrame(y_pred_cat, columns=TARGETS_CATEGORY)
            for c in TARGETS_CATEGORY: df_final[c] = df_res_cat[c]
            
        except Exception as e: return None, f"Error Model Category: {e}"

    # 5. EXPORT DENGAN LAYOUT KHUSUS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"RESULT_{report_type.upper()}_{timestamp}.xlsx"
    output_path = os.path.join("datasets", output_filename)
    
    try:
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        sheet_name = 'Defect Classification' if report_type == 'daily' else 'OZ,MS,IH CATEGORY'
        
        apply_custom_layout(writer, df_final, sheet_name, report_type)
        
        writer.close()
    except Exception as e:
        return None, f"Gagal styling excel: {str(e)}"

    return output_path, "Success"