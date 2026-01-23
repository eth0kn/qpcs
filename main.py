from fastapi import FastAPI, UploadFile, File
import tempfile
import os
import threading

from training_script import (
    train_ai_advanced,
    predict_excel_process
)

app = FastAPI(title="QPCS AI Backend")

# =============================================================================
# TRAIN STATE
# =============================================================================
training_state = {
    "running": False,
    "progress": 0,
    "message": ""
}


def progress_callback(p, m):
    training_state["progress"] = p
    training_state["message"] = m


# =============================================================================
# HEALTH
# =============================================================================
@app.get("/")
def health():
    return {"status": "ok"}


# =============================================================================
# TRAIN (THREAD SAFE)
# =============================================================================
@app.post("/train")
def train():
    if training_state["running"]:
        return {"status": "already running"}

    def run():
        training_state["running"] = True
        try:
            train_ai_advanced(progress_callback=progress_callback)
        finally:
            training_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return {"status": "training started"}


@app.get("/progress")
def progress():
    return training_state


# =============================================================================
# PREDICT
# =============================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return predict_excel_process(tmp_path)
    finally:
        os.unlink(tmp_path)
