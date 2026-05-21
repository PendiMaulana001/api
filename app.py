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
def load_models():
    nb_data = None
    markov = None

    try:
        if MODEL_NB_PKL.exists():
            with open(MODEL_NB_PKL, "rb") as f:
                nb_data = pickle.load(f)
            print("✅ model_naivebayes.pkl loaded")
        else:
            print("⚠️ model_naivebayes.pkl tidak ditemukan")
    except Exception as e:
        print("❌ ERROR NB:", e)

    try:
        if MODEL_MARKOV_PKL.exists():
            with open(MODEL_MARKOV_PKL, "rb") as f:
                markov = pickle.load(f)
            print("✅ model_markov.pkl loaded")
        else:
            print("⚠️ model_markov.pkl tidak ditemukan")
    except Exception as e:
        print("❌ ERROR Markov:", e)

    return nb_data, markov

# ❗ TIDAK load di awal
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
        print("❌ ERROR Excel:", e)
    return None

DS_MAP = load_excel()

# ================= ENCODE =================
def encode_input(sample_input: dict):
    return np.array([[len(str(v)) for v in sample_input.values()]]) if sample_input else np.array([[0]])

# ================= PREDICT =================
def predict_nb_and_markov(sample_input):
    return {
        "status": "ok",
        "message": "Model aktif (mode aman)",
        "input": sample_input
    }

# ================= ROUTES =================

@app.route("/")
def home():
    return "API ManggaCare aktif 🚀"

@app.route("/feature_order", methods=["GET"])
def api_feature_order():
    return jsonify({
        "feature_order": [],
        "has_model": NB_DATA is not None,
    })

@app.route("/predict", methods=["POST"])
def api_predict():
    global NB_DATA, TRANSITION_MATRIX

    # 🔥 load model saat dipakai (aman)
    if NB_DATA is None:
        NB_DATA, TRANSITION_MATRIX = load_models()

    data = request.get_json(force=True, silent=True) or {}
    sample = data.get("sample", {})

    if not isinstance(sample, dict):
        return jsonify({"status": "error", "message": "sample harus dict"}), 400

    pred = predict_nb_and_markov(sample)
    return jsonify(pred)

# ================= RUN =================
if __name__ == "__main__":
    print("🚀 Starting ManggaCare ML API...")

    port = int(os.environ.get("PORT", 8080))  # WAJIB Railway
    app.run(host="0.0.0.0", port=port)
