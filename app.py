"""
Exoplanet Verification System — Flask Backend
==============================================
Loads trained models from kepler_analysis.ipynb pipeline and serves predictions.

Prediction flow:
  1. Accept user inputs (7 required + 3 optional)
  2. Auto-derive koi_teq, koi_insol if missing
  3. Regression: predict koi_prad (planetary radius) using 8 regressors
  4. Compute engineered features from predicted koi_prad
  5. Classification: predict CONFIRMED vs FALSE POSITIVE using 8 classifiers
  6. Habitability assessment
  7. Return structured JSON
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ─── Load metadata & models ──────────────────────────────────────────────────
with open(META_PATH) as f:
    META = json.load(f)

CLF_FEATURES = META["clf_feature_cols"]
REG_FEATURES = META["reg_feature_cols"]
MEDIANS = META["medians"]

clf_scaler = joblib.load(os.path.join(MODEL_DIR, "clf_scaler.joblib"))
reg_scaler = joblib.load(os.path.join(MODEL_DIR, "reg_scaler.joblib"))

clf_models = []
for info in META["clf_models"]:
    model = joblib.load(os.path.join(MODEL_DIR, info["file"]))
    clf_models.append({**info, "model": model})

reg_models = []
for info in META["reg_models"]:
    model = joblib.load(os.path.join(MODEL_DIR, info["file"]))
    reg_models.append({**info, "model": model})

print(f"  Loaded {len(clf_models)} classifiers, {len(reg_models)} regressors")

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ─── Helper: auto-derive equilibrium temperature ─────────────────────────────
def _auto_teq(steff, srad, period):
    """Rough estimate: T_eq ≈ T_star × √(R_star_AU / (2 × a_AU))"""
    P_yr = period / 365.25
    M_star = max(srad, 0.1)  # rough M–R for main sequence
    a_AU = max((P_yr ** 2 * M_star) ** (1 / 3), 0.01)
    R_star_AU = srad * 0.00465047  # R_sun in AU
    return steff * (R_star_AU / (2.0 * a_AU)) ** 0.5


def _auto_insol(steff, srad, period):
    """Insolation flux relative to Earth: (T/T_sun)^4 × (R/a)^2"""
    P_yr = period / 365.25
    M_star = max(srad, 0.1)
    a_AU = max((P_yr ** 2 * M_star) ** (1 / 3), 0.01)
    return (steff / 5778.0) ** 4 * (srad / a_AU) ** 2


def _float(v):
    """Safe float parse."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ─── Prediction pipeline ─────────────────────────────────────────────────────
def predict(data):
    # ── Parse required inputs ──
    koi_period   = _float(data.get("koi_period"))
    koi_depth    = _float(data.get("koi_depth"))
    koi_duration = _float(data.get("koi_duration"))
    koi_steff    = _float(data.get("koi_steff"))
    koi_srad     = _float(data.get("koi_srad"))

    required = {
        "Orbital Period (days)": koi_period,
        "Transit Depth (ppm)": koi_depth,
        "Transit Duration (hrs)": koi_duration,
        "Stellar Eff. Temp (K)": koi_steff,
        "Stellar Radius (R☉)": koi_srad,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}

    # ── Optional inputs (auto-derive / default if not given) ──
    ra = _float(data.get("ra"))
    if ra is None:
        ra = MEDIANS.get("ra", 291.0)
    dec = _float(data.get("dec"))
    if dec is None:
        dec = MEDIANS.get("dec", 44.0)

    koi_score = _float(data.get("koi_score"))
    if koi_score is None:
        koi_score = MEDIANS.get("koi_score", 0.5)

    koi_ror = _float(data.get("koi_ror"))  # planet/star radius ratio

    koi_teq = _float(data.get("koi_teq"))
    teq_auto = koi_teq is None
    if teq_auto:
        koi_teq = _auto_teq(koi_steff, koi_srad, koi_period)

    koi_insol = _float(data.get("koi_insol"))
    insol_auto = koi_insol is None
    if insol_auto:
        koi_insol = _auto_insol(koi_steff, koi_srad, koi_period)

    # ── Apply IQR capping (FOR ML MODELS ONLY — raw values kept for physics) ──
    bounds = META.get("iqr_bounds", {})
    def _cap(val, col):
        if col in bounds:
            return max(bounds[col]["lower"], min(val, bounds[col]["upper"]))
        return val

    # Store RAW values for physics & habitability (before capping)
    raw_period   = koi_period
    raw_depth    = koi_depth
    raw_duration = koi_duration
    raw_srad     = koi_srad
    raw_steff    = koi_steff

    # Capped values for ML model inputs only
    koi_score    = _cap(koi_score, "koi_score")
    koi_period   = _cap(koi_period, "koi_period")
    koi_teq      = _cap(koi_teq, "koi_teq")
    koi_insol    = _cap(koi_insol, "koi_insol")
    koi_depth    = _cap(koi_depth, "koi_depth")
    koi_duration = _cap(koi_duration, "koi_duration")
    koi_srad     = _cap(koi_srad, "koi_srad")
    koi_steff    = _cap(koi_steff, "koi_steff")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: DETERMINE PLANETARY RADIUS
    # ══════════════════════════════════════════════════════════════════════
    user_prad = _float(data.get("koi_prad"))  # optional user-provided radius
    period_temp = koi_period * koi_teq

    R_SUN_IN_REARTH = 109.076

    # If koi_ror (planet/star radius ratio) is given, use it directly —
    # it's the most accurate way to derive planet radius from transit data.
    if koi_ror is not None and koi_ror > 0:
        ror_prad = round(koi_ror * raw_srad * R_SUN_IN_REARTH, 4)
    else:
        ror_prad = None

    # Physics-based estimate using RAW (uncapped) depth & stellar radius
    # R_p = R_star × √(depth/1e6) × 109.076 R⊕/R☉
    depth_prad = round(raw_srad * (raw_depth / 1e6) ** 0.5 * R_SUN_IN_REARTH, 4)
    depth_prad = max(depth_prad, 0.01)

    # Prefer ror-based radius (direct measurement) over depth-based (approximation)
    physics_prad = ror_prad if ror_prad is not None else depth_prad

    # ML regression predictions (supplementary)
    reg_vals = {
        "koi_score": koi_score,
        "koi_period": koi_period,
        "koi_teq": koi_teq,
        "koi_insol": koi_insol,
        "koi_depth": koi_depth,
        "koi_duration": koi_duration,
        "koi_srad": koi_srad,
        "koi_steff": koi_steff,
        "ra": ra,
        "dec": dec,
        "period_temp_product": period_temp,
    }
    reg_vec = np.array([[reg_vals.get(f, MEDIANS.get(f, 0)) for f in REG_FEATURES]])
    reg_vec_scaled = reg_scaler.transform(reg_vec)

    reg_predictions = {}
    for rm in reg_models:
        inp = reg_vec_scaled if rm["scaled"] else reg_vec
        pred = float(rm["model"].predict(inp)[0])
        reg_predictions[rm["name"]] = round(max(pred, 0.01), 4)

    # Use only tree-based models for ML avg (linear models often predict ≤0 for small planets)
    tree_models = ["Random Forest", "Gradient Boosting", "LightGBM"]
    tree_preds = [v for k, v in reg_predictions.items() if k in tree_models]
    ml_avg_prad = round(np.mean(tree_preds) if tree_preds else np.mean(list(reg_predictions.values())), 4)

    # Final radius determination:
    #   1. If user provided koi_prad → use it (known planet)
    #   2. Otherwise → weighted blend: 85% physics + 15% ML (tree models)
    #      Physics is consistently most accurate (< 2% error for known planets).
    #      ML provides supplementary signal from learned population statistics.
    if user_prad is not None and user_prad > 0:
        predicted_prad = round(user_prad, 4)
        radius_source = "user"
    else:
        predicted_prad = round(0.85 * physics_prad + 0.15 * ml_avg_prad, 4)
        radius_source = "estimated"

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: CLASSIFICATION — Is it CONFIRMED?
    # ══════════════════════════════════════════════════════════════════════
    psr_ratio_final = predicted_prad / max(raw_srad, 0.01)
    koi_prad_capped = _cap(predicted_prad, "koi_prad")

    clf_vals = {
        "koi_score": koi_score,
        "koi_period": koi_period,
        "koi_teq": koi_teq,
        "koi_insol": koi_insol,
        "koi_depth": koi_depth,
        "koi_duration": koi_duration,
        "koi_prad": koi_prad_capped,
        "koi_srad": koi_srad,
        "koi_steff": koi_steff,
        "ra": ra,
        "dec": dec,
        "planet_star_radius_ratio": psr_ratio_final,
        "period_temp_product": period_temp,
    }
    clf_vec = np.array([[clf_vals.get(f, MEDIANS.get(f, 0)) for f in CLF_FEATURES]])
    clf_vec_scaled = clf_scaler.transform(clf_vec)

    clf_predictions = {}
    for cm in clf_models:
        inp = clf_vec_scaled if cm["scaled"] else clf_vec
        prob = float(cm["model"].predict_proba(inp)[0][1]) * 100
        clf_predictions[cm["name"]] = round(prob, 2)

    # Primary prediction from Voting Ensemble
    ensemble_prob = clf_predictions.get("Voting Ensemble", 50.0)
    is_confirmed = ensemble_prob >= 50.0

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: HABITABILITY ASSESSMENT (uses raw physical values)
    # ══════════════════════════════════════════════════════════════════════
    # Use user-supplied or auto-derived T_eq/insol (NOT capped versions)
    hab_teq = _float(data.get("koi_teq"))
    if hab_teq is None:
        hab_teq = _auto_teq(raw_steff, raw_srad, raw_period)
    hab_insol = _float(data.get("koi_insol"))
    if hab_insol is None:
        hab_insol = _auto_insol(raw_steff, raw_srad, raw_period)

    hab_score = 0
    hab_factors = []

    if 180 <= hab_teq <= 310:
        hab_score += 30
        hab_factors.append("Temperature in habitable range (180–310 K)")
    elif 150 <= hab_teq <= 400:
        hab_score += 15
        hab_factors.append("Temperature near habitable range")

    if 0.8 <= hab_insol <= 1.5:
        hab_score += 25
        hab_factors.append("Insolation flux Earth-like (conservative HZ)")
    elif 0.25 <= hab_insol <= 2.2:
        hab_score += 12
        hab_factors.append("Insolation within optimistic habitable zone")

    if 0.5 <= predicted_prad <= 1.6:
        hab_score += 25
        hab_factors.append("Earth-like size — likely rocky")
    elif 1.6 < predicted_prad <= 2.5:
        hab_score += 10
        hab_factors.append("Super-Earth size — possibly rocky")

    if is_confirmed:
        hab_score += 15
        hab_factors.append("Classified as confirmed exoplanet")

    if 10.0 <= raw_period <= 500:
        hab_score += 5
        hab_factors.append("Orbital period allows stable climate")
    elif raw_period < 10.0:
        hab_factors.append("Ultra-short period — likely tidally locked")

    hab_score = min(hab_score, 100)

    # Planet type classification
    if predicted_prad < 0.8:
        ptype, pdesc = "Sub-Earth", "Smaller than Earth — likely rocky body"
    elif predicted_prad <= 1.25:
        ptype, pdesc = "Earth-like", "Similar to Earth — potential for habitability"
    elif predicted_prad <= 2.0:
        ptype, pdesc = "Super-Earth", "Larger than Earth — may have thick atmosphere"
    elif predicted_prad <= 4.0:
        ptype, pdesc = "Mini-Neptune", "Small gas/ice planet — less likely habitable"
    elif predicted_prad <= 10.0:
        ptype, pdesc = "Neptune-like", "Ice/gas giant — not habitable"
    else:
        ptype, pdesc = "Gas Giant", "Jupiter-class gas giant — not habitable"

    # ══════════════════════════════════════════════════════════════════════
    # BUILD RESPONSE
    # ══════════════════════════════════════════════════════════════════════
    return {
        "prediction": "CONFIRMED" if is_confirmed else "FALSE POSITIVE",
        "confirmation_probability": round(ensemble_prob, 2),
        "predicted_radius": predicted_prad,
        "radius_source": radius_source,
        "physics_radius": physics_prad,
        "ml_avg_radius": ml_avg_prad,
        "planet_type": ptype,
        "planet_type_desc": pdesc,
        "habitability_score": hab_score,
        "habitability_factors": hab_factors,
        "equilibrium_temp": round(hab_teq, 1),
        "insolation_flux": round(hab_insol, 4),
        "teq_auto_derived": teq_auto,
        "insol_auto_derived": insol_auto,
        "clf_breakdown": clf_predictions,
        "reg_breakdown": reg_predictions,
        "clf_metrics": {m["name"]: m["metrics"] for m in META["clf_models"]},
        "reg_metrics": {m["name"]: m["metrics"] for m in META["reg_models"]},
        "input_summary": {
            "koi_period": raw_period,
            "koi_depth": raw_depth,
            "koi_duration": raw_duration,
            "koi_steff": raw_steff,
            "koi_srad": raw_srad,
            "ra": ra,
            "dec": dec,
            "koi_score": round(koi_score, 4),
            "koi_teq": round(hab_teq, 1),
            "koi_insol": round(hab_insol, 4),
        },
        "dataset_info": META.get("dataset_info", {}),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    result = predict(data)
    return jsonify(result)


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """Return model metadata for frontend context."""
    return jsonify({
        "clf_models": [{"name": m["name"], "metrics": m["metrics"]} for m in META["clf_models"]],
        "reg_models": [{"name": m["name"], "metrics": m["metrics"]} for m in META["reg_models"]],
        "feature_stats": META.get("feature_stats", {}),
        "dataset_info": META.get("dataset_info", {}),
        "regression_target_stats": META.get("regression_target_stats", {}),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  Exoplanet Verification System")
    print(f"  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
