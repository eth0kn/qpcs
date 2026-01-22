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
    try:
        df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=30)
        for idx, row in df_temp.iterrows():
            row_str = row.astype(str).str.upper().tolist()
            if keyword.upper() in row_str:
                print(f"   > Header '{keyword}' ditemukan di baris ke-{idx + 1}")
                return idx
        return 1
    except Exception as e:
        print(f"   ! Error cari header: {str(e)}")
        return 1

def load_and_validate_data(file_path, sheet_name):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    header_idx = find_header_index(file_path, sheet_name, HEADER_KEYWORD)
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")
        return pd.DataFrame()

    if HEADER_KEYWORD not in df.columns:
        return pd.DataFrame()

    df[HEADER_KEYWORD] = df[HEADER_KEYWORD].fillna("").astype(str)
    # Filter data kosong/tidak valid
    df = df[df[HEADER_KEYWORD].str.strip().str.len() > 1]
    
    return df

# ==============================================================================
# AI EMBEDDER (OPTIMIZED)
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
# STYLING ENGINE (XlsxWriter)
# ==============================================================================
def apply_professional_style(writer, df, sheet_name, report_type):
    """
    Fungsi ini akan melukis Excel agar rapi, berwarna, dan professional.
    """
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # 1. DEFINISI STYLE
    # Header: Biru Tua, Teks Putih Bold
    header_fmt = workbook.add_format({
        'bold': True, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
        'fg_color': '#203764', 'font_color': 'white', 'border': 1
    })
    
    # Body Normal: Border tipis, text wrap (untuk deskripsi panjang)
    body_fmt = workbook.add_format({
        'border': 1, 'valign': 'top', 'text_wrap': True, 'font_size': 11
    })
    
    # Body Center: Untuk NO, Kode Symptom (agar rapi ditengah)
    center_fmt = workbook.add_format({
        'border': 1, 'valign': 'top', 'align': 'center', 'font_size': 11
    })
    
    # Highlight AI (Kuning Muda): Untuk kolom hasil prediksi
    ai_fmt = workbook.add_format({
        'border': 1, 'valign': 'top', 'text_wrap': True, 
        'fg_color': '#FFFFCC', 'font_size': 11, 'bold': False
    })

    # 2. SET LEBAR KOLOM & FORMAT
    # Loop semua kolom untuk menentukan format
    for col_num, col_name in enumerate(df.columns):
        col_name_upper = str(col_name).upper()
        
        # A. Tentukan Lebar Kolom
        if col_name_upper == "NO":
            width = 5
            style = center_fmt
        elif "SYMPTOM" in col_name_upper and "DESC" not in col_name_upper: # Kode Symptom
            width = 10
            style = center_fmt
        elif "DESC" in col_name_upper or "DETAIL" in col_name_upper or "REMARK" in col_name_upper:
            width = 45 # Kolom Deskripsi Lebar
            style = body_fmt
        elif "DEFECT" in col_name_upper or "SVC TYPE" in col_name_upper or "REASON" in col_name_upper:
            width = 20
            style = ai_fmt # Pakai warna highlight AI
        else:
            width = 15
            style = body_fmt

        # B. Terapkan Style ke Kolom
        # set_column(first_col, last_col, width, cell_format)
        worksheet.set_column(col_num, col_num, width, style)
        
        # C. Tulis Ulang Header dengan Style Header
        worksheet.write(0, col_num, col_name, header_fmt)

    # 3. FREEZE HEADER & FILTER
    worksheet.freeze_panes(1, 0) # Bekukan baris 1
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1) # Aktifkan Filter

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    global report
    if progress_callback:
        def report(p, m):
            progress_callback(p, m)
            print(f"[{p}%] {m}")

    report(10, "Menganalisa Struktur File & Load Data...")

    df_defect = load_and_validate_data(DATASET_PATH, SHEET_DAILY)
    df_oz = load_and_validate_data(DATASET_PATH, SHEET_MONTHLY)

    targets_defect = ['Defect1', 'Defect2', 'Defect3']
    targets_oz = ['SVC TYPE', 'DETAIL REASON']

    # --- TRAIN MODEL DEFECT ---
    if not df_defect.empty:
        report(20, f"Training Model Defect ({len(df_defect)} data)...")
        X = df_defect[HEADER_KEYWORD].tolist()
        y = df_defect[targets_defect].fillna("-")
        
        pipeline = Pipeline([
            ('embedder', BertEmbedder(MODEL_NAME)),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))
        ])
        
        pipeline.fit(X, y)
        report(40, "Menyimpan Model Defect...")
        joblib.dump(pipeline, f'{MODEL_DIR}model_defect.pkl')
        del pipeline, X, y
        gc.collect()
    else:
        report(20, "Skip Model Defect (Data Kosong)")

    # --- TRAIN MODEL CATEGORY ---
    if not df_oz.empty:
        report(60, f"Training Model Category ({len(df_oz)} data)...")
        X = df_oz[HEADER_KEYWORD].tolist()
        y = df_oz[targets_oz].fillna("-")
        
        pipeline_oz = Pipeline([
            ('embedder', BertEmbedder(MODEL_NAME)),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=-1)))
        ])
        
        pipeline_oz.fit(X, y)
        report(80, "Menyimpan Model Category...")
        joblib.dump(pipeline_oz, f'{MODEL_DIR}model_oz.pkl')
        del pipeline_oz, X, y
        gc.collect()
    else:
        report(60, "Skip Model Category (Data Kosong)")

    report(100, "Training Selesai.")

def predict_excel_process(input_file_path, report_type='daily'):
    import datetime
    
    # 1. Load Model
    try:
        if report_type == 'daily':
            model_path = f'{MODEL_DIR}model_defect.pkl'
            # Urutan kolom sesuai format yang diminta user
            targets = ['Defect1', 'Defect2', 'Defect3']
        else: 
            model_path = f'{MODEL_DIR}model_oz.pkl'
            targets = ['SVC TYPE', 'DETAIL REASON']
            
        if not os.path.exists(model_path):
            return None, "Model belum ditraining."

        pipeline = joblib.load(model_path)
            
    except Exception as e:
        return None, str(e)

    # 2. Load Data
    try:
        xls = pd.ExcelFile(input_file_path)
        sheet_name = xls.sheet_names[0]
        header_idx = find_header_index(input_file_path, sheet_name, HEADER_KEYWORD)
        df = pd.read_excel(input_file_path, sheet_name=sheet_name, header=header_idx)
        
        if HEADER_KEYWORD not in df.columns:
            return None, f"Kolom '{HEADER_KEYWORD}' tidak ditemukan."
            
    except Exception as e:
        return None, f"Gagal baca file: {str(e)}"

    # 3. Prediksi
    X_pred = df[HEADER_KEYWORD].fillna("").astype(str).tolist()
    
    try:
        y_pred = pipeline.predict(X_pred)
    except Exception as e:
        return None, f"Error prediksi: {str(e)}"

    # 4. Susun Hasil ke DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    
    # Masukkan hasil prediksi ke dataframe utama
    for col in targets:
        df[col] = y_pred_df[col]
        
    # Tambahkan kolom RESULT (Kosong) jika Daily, sesuai format
    if report_type == 'daily' and 'RESULT' not in df.columns:
        df['RESULT'] = ""

    # 5. Export dengan Styling (XlsxWriter)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"RESULT_{report_type}_{timestamp}.xlsx"
    output_path = os.path.join("datasets", output_filename)
    
    try:
        # Gunakan engine xlsxwriter
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Result')
        
        # Panggil fungsi styling
        apply_professional_style(writer, df, 'Result', report_type)
        
        writer.close()
    except Exception as e:
        return None, f"Gagal styling excel: {str(e)}"

    return output_path, "Success"