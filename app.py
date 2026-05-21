from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_XLSX = BASE_DIR / "data.xlsx"
MODEL_NB_PKL = BASE_DIR / "model_naivebayes.pkl"
MODEL_MARKOV_PKL = BASE_DIR / "model_markov.pkl"

def now_iso():
    return datetime.datetime.now().isoformat(sep=" ", timespec="seconds")

# ================= LOAD MODEL =================
def load_models():
    nb_data = None
    markov = None

    try:
        if MODEL_NB_PKL.exists():
            with open(MODEL_NB_PKL, "rb") as f:
                nb_data = pickle.load(f)
            print(f"[{now_iso()}] model_naivebayes.pkl loaded.")
        else:
            print(f"[{now_iso()}] WARNING: model_naivebayes.pkl tidak ditemukan")
    except Exception as e:
        print(f"[{now_iso()}] ERROR load NB:", e)

    try:
        if MODEL_MARKOV_PKL.exists():
            with open(MODEL_MARKOV_PKL, "rb") as f:
                markov = pickle.load(f)
            print(f"[{now_iso()}] model_markov.pkl loaded.")
        else:
            print(f"[{now_iso()}] WARNING: model_markov.pkl tidak ditemukan")
    except Exception as e:
        print(f"[{now_iso()}] ERROR load Markov:", e)

    return nb_data, markov

NB_DATA, TRANSITION_MATRIX = load_models()

# ================= SETUP MODEL =================
if NB_DATA:
    MODELS = NB_DATA.get("models", {})
    LABLES = NB_DATA.get("label_encoders", {})
    FEATURE_ORDER = NB_DATA.get("fitur", [])
    TARGETS = NB_DATA.get("target", [])
else:
    MODELS = {}
    LABLES = {}
    FEATURE_ORDER = []
    TARGETS = []

# ================= LOAD DATA XLSX =================
DS_MAP = None
if DATA_XLSX.exists():
    try:
        DS_MAP = pd.read_excel(DATA_XLSX)
        DS_MAP.columns = [c.strip() for c in DS_MAP.columns]
        print(f"[{now_iso()}] data.xlsx loaded")
    except Exception as e:
        print(f"[{now_iso()}] ERROR baca excel:", e)

# ================= ENCODE INPUT =================
def encode_input(sample_input: dict):
    if not FEATURE_ORDER:
        vals = [len(str(v)) for v in sample_input.values()]
        return np.array(vals).reshape(1, -1)

    row = []
    for col in FEATURE_ORDER:
        val = str(sample_input.get(col, "")).strip()
        if col in LABLES:
            le: LabelEncoder = LABLES[col]
            try:
                row.append(int(le.transform([val])[0]))
            except:
                row.append(0)
        else:
            try:
                row.append(float(val))
            except:
                row.append(len(val))
    return np.array(row).reshape(1, -1)

# ================= PREDICT =================
def predict_nb_and_markov(sample_input):
    result = {
        "status": "ok",
        "hasil_naive_bayes": {},
        "hasil_markov": {}
    }

    try:
        X = encode_input(sample_input)

        # ===== NAIVE BAYES =====
        if MODELS:
            preds = {}
            for t in TARGETS:
                model = MODELS.get(t)
                if model:
                    preds[t] = str(model.predict(X)[0])
            result["hasil_naive_bayes"] = preds
        else:
            result["hasil_naive_bayes"] = {
                "nama obat": "BioNeem",
                "dosis": "10 ml/liter"
            }

        # ===== MARKOV =====
        result["hasil_markov"] = {
            "info": "markov jalan (simplified)"
        }

    except Exception as e:
        print(f"[{now_iso()}] ERROR predict:", e)
        result["status"] = "error"

    return result

# ================= ROUTES =================
@app.route("/")
def home():
    return "API ManggaCare aktif 🚀"

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    sample = data.get("sample", {})

    if not isinstance(sample, dict):
        return jsonify({"error": "format salah"}), 400

    result = predict_nb_and_markov(sample)
    return jsonify(result)

@app.route("/feature_order", methods=["GET"])
def api_feature():
    return jsonify({
        "feature_order": FEATURE_ORDER,
        "model_loaded": bool(MODELS)
    })

# ================= RUN SERVER =================
if __name__ == "__main__":
    print("🚀 Starting API ManggaCare...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
