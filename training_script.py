import pandas as pd
import joblib
import os
import numpy as np
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

# CONFIG SHEET NAMES
SHEET_DAILY = "PROCESS (DEFECT)"
SHEET_MONTHLY = "PROCESS (OZ,MS,IH)"

# Kunci untuk Auto Detect Header
HEADER_KEYWORD = "PROC_DETAIL_E"

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def report(progress, message):
    """
    Fungsi helper untuk print status. 
    Nanti di Tahap 2 akan kita sambungkan ke Frontend.
    """
    print(f"[PROGRESS {progress}%] {message}")
    # Placeholder untuk callback status ke main.py jika diperlukan nanti

def find_header_index(file_path, sheet_name, keyword):
    """
    Mencari index baris dimana terdapat keyword tertentu (PROC_DETAIL_E).
    Mencari max di 30 baris pertama untuk efisiensi.
    """
    try:
        # Baca tanpa header dulu, ambil 30 baris saja
        df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=30)
        
        for idx, row in df_temp.iterrows():
            # Convert row jadi string lalu cari keyword (case insensitive)
            row_str = row.astype(str).str.upper().tolist()
            if keyword.upper() in row_str:
                print(f"   > Header '{keyword}' ditemukan di baris ke-{idx + 1} pada sheet '{sheet_name}'")
                return idx
                
        print(f"   ! Warning: Header '{keyword}' tidak ditemukan di 30 baris awal sheet '{sheet_name}'. Menggunakan default baris 1.")
        return 1 # Default fallback
    except Exception as e:
        print(f"   ! Error saat mencari header: {str(e)}")
        return 1

def load_and_validate_data(file_path, sheet_name):
    """
    Load data dengan dynamic header dan membuang baris invalid (NaN di PROC_DETAIL_E).
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    # 1. Cari Header Position
    header_idx = find_header_index(file_path, sheet_name, HEADER_KEYWORD)
    
    # 2. Load Data Real
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")
        return pd.DataFrame()

    # 3. Validasi Kolom Kunci
    if HEADER_KEYWORD not in df.columns:
        print(f"Kolom {HEADER_KEYWORD} tidak ditemukan di sheet {sheet_name}. Kolom tersedia: {list(df.columns)}")
        return pd.DataFrame()

    initial_count = len(df)

    # 4. Data Cleaning Minimal (Hanya memastikan tipe data String)
    # Convert NaN menjadi string kosong dulu agar bisa dicek
    df[HEADER_KEYWORD] = df[HEADER_KEYWORD].fillna("").astype(str)

    # 5. Filtering Logic (Poin B User)
    # Hapus jika kosong, whitespace saja, 'nan', atau '-'
    # Kita anggap data valid jika panjang string > 1 setelah di strip
    df = df[df[HEADER_KEYWORD].str.strip().str.len() > 1]
    
    # Hapus jika isinya cuma angka/simbol tidak jelas (opsional, tapi user minta skip yg tidak valid)
    # Disini kita pastikan row yang mau ditraining memiliki Label/Target
    
    final_count = len(df)
    dropped_count = initial_count - final_count
    
    if dropped_count > 0:
        print(f"   > Dibuang {dropped_count} baris invalid (PROC_DETAIL_E kosong/rusak) dari {sheet_name}")

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
        # Lazy loading model
        if self.model is None:
            print(f"   ...Loading Model BERT: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device='cpu') # Ubah ke 'cuda' jika ada GPU
        
        # Pastikan input adalah list string, bukan NaN
        X = [str(text) if text is not None else "" for text in X]
        
        print(f"   ...Encoding {len(X)} kalimat...")
        return self.model.encode(X, show_progress_bar=True)

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_ai_advanced(enable_cleansing=False, progress_callback=None):
    """
    Main function yang dipanggil oleh main.py
    Param enable_cleansing kita ignore karena user minta RAW DATA.
    """
    global report
    if progress_callback:
        def report(p, m):
            progress_callback(p, m)
            print(f"[{p}%] {m}")

    report(10, "Menganalisa Struktur File Excel...")

    # --- 1. LOAD DATASET (DEFECT / DAILY) ---
    df_defect = load_and_validate_data(DATASET_PATH, SHEET_DAILY)
    
    # --- 2. LOAD DATASET (CATEGORY / MONTHLY) ---
    df_oz = load_and_validate_data(DATASET_PATH, SHEET_MONTHLY)

    # Definisi Target Columns (Sesuaikan dengan CSV user: Defect1, Defect2, Defect3)
    # Menggunakan Title Case sesuai file yang dikirim user
    targets_defect = ['Defect1', 'Defect2', 'Defect3']
    targets_oz = ['SVC TYPE', 'DETAIL REASON']

    # --- 3. TRAIN MODEL DEFECT ---
    if not df_defect.empty:
        report(20, f"Training Model Defect ({len(df_defect)} data)...")
        
        # Siapkan X dan Y
        X = df_defect[HEADER_KEYWORD].tolist() # Raw Text
        y = df_defect[targets_defect].fillna("-") # Isi label kosong dengan -
        
        # Pipeline: Embedder -> Classifier
        # Menggunakan parameter Random Forest standard yang seimbang
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
        report(20, "Skip Model Defect (Data Kosong/Tidak Valid)")

    # --- 4. TRAIN MODEL CATEGORY (OZ) ---
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
        report(60, "Skip Model Category (Data Kosong/Tidak Valid)")

    report(100, "Training Selesai.")

# ==============================================================================
# TAMBAHAN TAHAP 3: PREDICTION LOGIC
# ==============================================================================

def predict_excel_process(input_file_path, report_type='daily'):
    """
    Fungsi untuk memproses file excel, mencari header otomatis,
    melakukan prediksi tanpa cleansing, dan mengembalikan path file hasil.
    """
    import datetime
    
    # 1. Load Model yang Sesuai
    try:
        if report_type == 'daily':
            model_path = f'{MODEL_DIR}model_defect.pkl'
            targets = ['Defect1', 'Defect2', 'Defect3']
            if not os.path.exists(model_path):
                raise Exception("Model Defect belum mentraining. Silakan training dulu.")
            pipeline = joblib.load(model_path)
            
        else: # monthly
            model_path = f'{MODEL_DIR}model_oz.pkl'
            targets = ['SVC TYPE', 'DETAIL REASON']
            if not os.path.exists(model_path):
                raise Exception("Model Category (OZ) belum mentraining. Silakan training dulu.")
            pipeline = joblib.load(model_path)
            
    except Exception as e:
        return None, str(e)

    # 2. Cari Header & Load Data
    # Kita gunakan sheet pertama secara default untuk prediksi
    try:
        xls = pd.ExcelFile(input_file_path)
        sheet_name = xls.sheet_names[0] # Ambil sheet pertama
        
        header_idx = find_header_index(input_file_path, sheet_name, HEADER_KEYWORD)
        
        # Load data, keep original untuk ditulis ulang nanti
        df = pd.read_excel(input_file_path, sheet_name=sheet_name, header=header_idx)
        
        if HEADER_KEYWORD not in df.columns:
            return None, f"Kolom '{HEADER_KEYWORD}' tidak ditemukan di sheet '{sheet_name}'"
            
    except Exception as e:
        return None, f"Gagal membaca file Excel: {str(e)}"

    # 3. Pre-processing (RAW - Tanpa Cleansing)
    # Convert ke string & handle NaN
    X_pred = df[HEADER_KEYWORD].fillna("").astype(str).tolist()
    
    # 4. Lakukan Prediksi
    try:
        # Pipeline akan otomatis melakukan embedding (Bert) -> classification
        y_pred = pipeline.predict(X_pred)
    except Exception as e:
        return None, f"Error saat prediksi AI: {str(e)}"

    # 5. Masukkan Hasil ke DataFrame
    # Pastikan jumlah baris sama
    if len(y_pred) != len(df):
        # Fallback jika ada mismatch (jarang terjadi)
        print("Warning: Size mismatch prediction")
    
    # Assign hasil ke kolom target
    y_pred_df = pd.DataFrame(y_pred, columns=targets)
    
    for col in targets:
        df[col] = y_pred_df[col]

    # 6. Simpan File Hasil
    # Format nama file hasil
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"RESULT_{report_type}_{timestamp}.xlsx"
    output_path = os.path.join("datasets", output_filename) # Simpan di folder datasets sementara
    
    try:
        df.to_excel(output_path, index=False)
    except Exception as e:
        return None, f"Gagal menyimpan file hasil: {str(e)}"
        
    return output_path, "Success"

if __name__ == "__main__":
    # Untuk testing manual via CLI
    train_ai_advanced()