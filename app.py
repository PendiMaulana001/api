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

# ================= SAFE LOAD =================
def safe_load_models():
    try:
        nb_data = None
        markov = None

        if MODEL_NB_PKL.exists():
            with open(MODEL_NB_PKL, "rb") as f:
                nb_data = pickle.load(f)
            print("NB loaded")
        else:
            print("NB file tidak ada")

        if MODEL_MARKOV_PKL.exists():
            with open(MODEL_MARKOV_PKL, "rb") as f:
                markov = pickle.load(f)
            print("Markov loaded")
        else:
            print("Markov file tidak ada")

        return nb_data, markov

    except Exception as e:
        print("ERROR LOAD MODEL:", e)
        return None, None

# ❗ JANGAN crash saat start
NB_DATA, TRANSITION_MATRIX = None, None

# ================= LOAD DATA (SAFE) =================
def load_excel():
    try:
        if DATA_XLSX.exists():
            df = pd.read_excel(DATA_XLSX)
            df.columns = [c.strip() for c in df.columns]
            return df
    except Exception as e:
        print("ERROR EXCEL:", e)
    return None

DS_MAP = load_excel()

# ================= ROUTES =================

@app.route("/")
def home():
    return "API ManggaCare aktif 🚀"

@app.route("/feature_order")
def feature():
    return jsonify({
        "feature_order": [],
        "model_loaded": NB_DATA is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    global NB_DATA, TRANSITION_MATRIX

    # 🔥 load model saat dipakai (bukan saat start)
    if NB_DATA is None:
        NB_DATA, TRANSITION_MATRIX = safe_load_models()

    data = request.get_json() or {}
    sample = data.get("sample", {})

    try:
        return jsonify({
            "status": "ok",
            "message": "API jalan + model dicoba load",
            "sample": sample,
            "model_loaded": NB_DATA is not None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        })

# ================= RUN =================
if __name__ == "__main__":
    print("🚀 Starting ManggaCare API...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
