from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import datetime
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

def load_models():
    nb_data = None
    markov = None

    try:
        if MODEL_NB_PKL.exists():
            with open(MODEL_NB_PKL, "rb") as f:
                nb_data = pickle.load(f)
            print(f"[{now_iso()}] model_naivebayes.pkl loaded.")
        else:
            print(f"[{now_iso()}] WARNING: model_naivebayes.pkl tidak ditemukan di: {MODEL_NB_PKL}")
    except Exception as e:
        print(f"[{now_iso()}] Warning: gagal load model_naivebayes.pkl ({e})")
        nb_data = None

    try:
        if MODEL_MARKOV_PKL.exists():
            with open(MODEL_MARKOV_PKL, "rb") as f:
                markov = pickle.load(f)
            print(f"[{now_iso()}] model_markov.pkl loaded.")
        else:
            print(f"[{now_iso()}] WARNING: model_markov.pkl tidak ditemukan di: {MODEL_MARKOV_PKL}")
            markov = None
    except Exception as e:
        print(f"[{now_iso()}] Warning: gagal load model_markov.pkl ({e})")
        markov = None

    return nb_data, markov

NB_DATA, TRANSITION_MATRIX = load_models()

if NB_DATA:
    MODELS = NB_DATA.get("models", {})
    LABLES = NB_DATA.get("label_encoders", {})
    FEATURE_ORDER = NB_DATA.get("fitur", [])
    TARGETS = NB_DATA.get("target", [])
    print(f"[{now_iso()}] MODELS loaded: {list(MODELS.keys())}")
    print(f"[{now_iso()}] FEATURE_ORDER: {FEATURE_ORDER}")
else:
    MODELS = {}
    LABLES = {}
    FEATURE_ORDER = []
    TARGETS = []
    print(f"[{now_iso()}] WARNING: NB_DATA kosong, MODELS tidak tersedia.")

DS_MAP = None
if DATA_XLSX.exists():
    try:
        DS_MAP = pd.read_excel(DATA_XLSX)
        DS_MAP.columns = [c.strip() for c in DS_MAP.columns]
        print(f"[{now_iso()}] data.xlsx loaded untuk mapping detail obat.")
    except Exception as e:
        print(f"[{now_iso()}] Warning: gagal baca {DATA_XLSX}: {e}")
        DS_MAP = None
else:
    print(f"[{now_iso()}] WARNING: data.xlsx tidak ditemukan di {DATA_XLSX}")

def encode_input(sample_input: dict):
    """
    Mengubah input dict (teks) menjadi vektor angka sesuai FEATURE_ORDER dan LABLES.
    """
    if not FEATURE_ORDER:
        vals = [len(str(v)) for v in sample_input.values() if v is not None]
        if not vals:
            vals = [0]
        return np.array(vals).reshape(1, -1)

    normalized = dict(sample_input)
    key_map = {
        "perkembangan": "perkembangan bunga atau buah",
        "kondisi_daun": "kondisi daun",
        "serangan_hama": "serangan hama",
        "tingkat_serangan": "tingkat serangan",
        "tingkat_kerontokan": "tingkat kerontokan buah atau bunga",
    }
    for k_src, k_dst in key_map.items():
        if k_src in normalized and k_dst not in normalized:
            normalized[k_dst] = normalized[k_src]

    row = []
    for col in FEATURE_ORDER:
        val = str(normalized.get(col, "")).strip()
        if col in LABLES:
            le: LabelEncoder = LABLES[col]
            classes = list(le.classes_)
            if val in classes:
                try:
                    row.append(int(le.transform([val])[0]))
                    continue
                except Exception:
                    pass
            lower_map = {c.lower(): c for c in classes}
            if val.lower() in lower_map:
                try:
                    row.append(int(le.transform([lower_map[val.lower()]])[0]))
                    continue
                except Exception:
                    pass
            try:
                row.append(int(le.transform([classes[0]])[0]))
                continue
            except Exception:
                row.append(0)
        else:
            try:
                row.append(float(val))
            except Exception:
                row.append(len(val))
    return np.array(row).reshape(1, -1)


def predict_outcome_after_treatment(sample_input, hasil_nb):
    tingkat_serangan = str(
        sample_input.get("tingkat serangan", sample_input.get("tingkat_serangan", ""))
    ).lower()
    tingkat_kerontokan = str(
        sample_input.get("tingkat kerontokan buah atau bunga",
                         sample_input.get("tingkat_kerontokan", ""))
    ).lower()

    def turun(level):
        if level == "tinggi":
            return "sedang"
        if level == "sedang":
            return "rendah"
        return level or "-"

    after_serangan = turun(tingkat_serangan)
    after_kerontokan = turun(tingkat_kerontokan)

    catatan = (
        f"Setelah diberikan obat {hasil_nb.get('nama obat','-')} dengan dosis "
        f"{hasil_nb.get('dosis obat','-')}, tingkat serangan diperkirakan turun "
        f"menjadi '{after_serangan}' dan kerontokan menjadi '{after_kerontokan}'."
    )

    return {
        "tingkat_serangan_sebelum": tingkat_serangan or "-",
        "tingkat_serangan_setelah": after_serangan or "-",
        "tingkat_kerontokan_sebelum": tingkat_kerontokan or "-",
        "tingkat_kerontokan_setelah": after_kerontokan or "-",
        "catatan": catatan,
    }

def predict_nb_and_markov(sample_input):
    result = {
        "status": "ok",
        "hasil_naive_bayes": None,
        "prediksi_setelah_naive_bayes": None,
        "hasil_markov": None,
    }

    X = encode_input(sample_input)
    if MODELS:
        try:
            preds = {}
            for t in TARGETS:
                model = MODELS.get(t)
                if model is None:
                    preds[t] = "-"
                    continue
                p_enc = model.predict(X)[0]
                if t in LABLES:
                    try:
                        p_lbl = LABLES[t].inverse_transform([int(p_enc)])[0]
                    except Exception:
                        p_lbl = str(p_enc)
                else:
                    p_lbl = str(p_enc)
                preds[t] = p_lbl

            hasil_nb = {
                "jenis obat": preds.get("jenis obat", "-"),
                "nama obat": preds.get("nama obat", "-"),
                "dosis obat": preds.get("dosis obat", "-"),
                "cara pakai": preds.get("cara pakai", "-"),
            }
            result["hasil_naive_bayes"] = hasil_nb
        except Exception as e:
            print(f"[{now_iso()}] NB error:", e)
            hasil_nb = {
                "jenis obat": "-",
                "nama obat": "-",
                "dosis obat": "-",
                "cara pakai": "-",
            }
            result["hasil_naive_bayes"] = hasil_nb
            result["status"] = "error"
    else:
        hasil_nb = {
            "jenis obat": "pestisida nabati",
            "nama obat": "BioNeem",
            "dosis obat": "10 ml/liter",
            "cara pakai": "semprot",
        }
        result["hasil_naive_bayes"] = hasil_nb
    result["prediksi_setelah_naive_bayes"] = predict_outcome_after_treatment(
        sample_input, hasil_nb
    )

    try:
        cur_nama = hasil_nb.get("nama obat", "-")
        cur_jenis = hasil_nb.get("jenis obat", "-")
        cur_dosis = hasil_nb.get("dosis obat", "-")
        cur_cara = hasil_nb.get("cara pakai", "-")

        if not TRANSITION_MATRIX:
            result["hasil_markov"] = {
                "nama obat": cur_nama,
                "prob": 1.0,
                "jenis obat": cur_jenis,
                "dosis obat": cur_dosis,
                "cara pakai": cur_cara,
            }
            return result

        le_name = LABLES.get("nama obat") or LABLES.get("nama_obat")
        if le_name is None:
            result["hasil_markov"] = {
                "nama obat": cur_nama,
                "prob": 1.0,
                "jenis obat": cur_jenis,
                "dosis obat": cur_dosis,
                "cara pakai": cur_cara,
            }
            return result
        
        enc_key = None
        try:
            enc_key = int(le_name.transform([str(cur_nama)])[0])
        except Exception:
            cs = {c.lower(): c for c in le_name.classes_}
            if str(cur_nama).lower() in cs:
                try:
                    enc_key = int(le_name.transform([cs[str(cur_nama).lower()]])[0])
                except Exception:
                    enc_key = None

        if enc_key is None:
            result["hasil_markov"] = {
                "nama obat": cur_nama,
                "prob": 1.0,
                "jenis obat": cur_jenis,
                "dosis obat": cur_dosis,
                "cara pakai": cur_cara,
            }
            return result

        best_next_idx = None
        best_prob = -1.0

        for src_state, dests in TRANSITION_MATRIX.items():
            src_idx = src_state
            if src_idx != enc_key:
                continue
            for dest_state, p in dests.items():
                dest_idx = dest_state
                p_float = float(p)
                if p_float > best_prob:
                    best_prob = p_float
                    best_next_idx = int(dest_idx)

        if best_next_idx is None:
            result["hasil_markov"] = {
                "nama obat": cur_nama,
                "prob": 1.0,
                "jenis obat": cur_jenis,
                "dosis obat": cur_dosis,
                "cara pakai": cur_cara,
            }
            return result

        try:
            next_name = le_name.inverse_transform([best_next_idx])[0]
        except Exception:
            next_name = str(best_next_idx)

        detail_jenis = cur_jenis
        detail_dosis = cur_dosis
        detail_cara = cur_cara

        if DS_MAP is not None and "nama obat" in DS_MAP.columns:
            subset = DS_MAP[DS_MAP["nama obat"].astype(str).str.lower() == str(next_name).lower()]
            if not subset.empty:
                if "jenis obat" in DS_MAP.columns:
                    try:
                        detail_jenis = str(subset["jenis obat"].mode()[0])
                    except Exception:
                        detail_jenis = str(subset["jenis obat"].iloc[0])
                if "dosis obat" in DS_MAP.columns:
                    try:
                        detail_dosis = str(subset["dosis obat"].mode()[0])
                    except Exception:
                        detail_dosis = str(subset["dosis obat"].iloc[0])
                if "cara pakai" in DS_MAP.columns:
                    try:
                        detail_cara = str(subset["cara pakai"].mode()[0])
                    except Exception:
                        detail_cara = str(subset["cara pakai"].iloc[0])

        if best_prob <= 0:
            best_prob = 1.0

        result["hasil_markov"] = {
            "nama obat": str(next_name),
            "prob": float(best_prob),
            "jenis obat": detail_jenis,
            "dosis obat": detail_dosis,
            "cara pakai": detail_cara,
        }

    except Exception as e:
        print(f"[{now_iso()}] Markov error:", e)
        result["hasil_markov"] = {
            "nama obat": hasil_nb.get("nama obat", "-"),
            "prob": 1.0,
            "jenis obat": hasil_nb.get("jenis obat", "-"),
            "dosis obat": hasil_nb.get("dosis obat", "-"),
            "cara pakai": hasil_nb.get("cara pakai", "-"),
        }

    return result

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    sample = data.get("sample", {})
    if not isinstance(sample, dict):
        return jsonify({"status": "error", "message": "sample harus dict"}), 400

    pred = predict_nb_and_markov(sample)
    return jsonify(pred)

@app.route("/feature_order", methods=["GET"])
def api_feature_order():
    return jsonify({
        "feature_order": FEATURE_ORDER,
        "has_model": bool(MODELS),
    })

if __name__ == "__main__":
    print("Starting ManggaCare ML API...")
    print("Model NB :", "OK" if MODELS else "NONE")
    print("Markov   :", "OK" if TRANSITION_MATRIX else "NONE")
    app.run(host="127.0.0.1", port=5000, debug=False)
