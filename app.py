from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime
import os
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "data.xlsx"
MODEL_NB_PKL = BASE_DIR / "model_naivebayes.pkl"
MODEL_MARKOV_PKL = BASE_DIR / "model_markov.pkl"

def now_iso():
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

# ================= SAFE LOAD MODEL =================
def safe_load_models():
    try:
        nb_data = None
        markov = None

        if MODEL_NB_PKL.exists():
            with open(MODEL_NB_PKL, "rb") as f:
                nb_data = pickle.load(f)
            print("✅ NB model loaded")
        else:
            print("⚠️ model_naivebayes.pkl tidak ditemukan")

        if MODEL_MARKOV_PKL.exists():
            with open(MODEL_MARKOV_PKL, "rb") as f:
                markov = pickle.load(f)
            print("✅ Markov model loaded")
        else:
            print("⚠️ model_markov.pkl tidak ditemukan")

        return nb_data, markov

    except Exception as e:
        print("❌ ERROR LOAD MODEL:", e)
        return None, None

# ❗ JANGAN load di awal (biar tidak crash)
NB_DATA = None
TRANSITION_MATRIX = None

# ================= LOAD EXCEL =================
def load_excel():
    try:
        if DATA_XLSX.exists():
            df = pd.read_excel(DATA_XLSX)
            df.columns = [c.strip() for c in df.columns]
            print("✅ Excel loaded")
            return df
    except Exception as e:
        print("❌ ERROR EXCEL:", e)
    return None

DS_MAP = load_excel()

# ================= ROOT =================
@app.route("/")
def home():
    return "API ManggaCare aktif 🚀"

# ================= FEATURE =================
@app.route("/feature_order")
def feature():
    return jsonify({
        "feature_order": [],
        "model_loaded": NB_DATA is not None
    })

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    global NB_DATA, TRANSITION_MATRIX

    # 🔥 Load model saat dipanggil (AMAN)
    if NB_DATA is None:
        NB_DATA, TRANSITION_MATRIX = safe_load_models()

    data = request.get_json() or {}
    sample = data.get("sample", {})

    return jsonify({
        "status": "ok",
        "message": "API aktif + model load saat request",
        "sample": sample,
        "model_loaded": NB_DATA is not None
    })

# ================= RUN =================
if __name__ == "__main__":
    print("🚀 Starting ManggaCare API...")

    port = int(os.environ.get("PORT", 8080))  # WAJIB untuk Railway
    app.run(host="0.0.0.0", port=port)
