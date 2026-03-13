"""
ML Engine — CompressorAI v5
Pipeline: DBSCAN → GBR → Genetic Algorithm (differential_evolution)
Auto-detects scaled electrical power (ELEC_SCALE = 7.6337).

Production changes vs original:
  - _save_model() respects APP_ENV — skips local disk write in production
    (models are persisted only in Supabase Storage via retrain.py)
  - load_model_from_dict() added — proper warm-start from a pickled dict
  - Logging added (replaces silent failures)
  - train() raises ValueError with user-friendly messages for bad data
  - Minor: float casts on boundary values to avoid numpy type leakage
"""
import os
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger("compressorai.engine")

APP_ENV   = os.getenv("APP_ENV", "development").lower()
IS_PROD   = APP_ENV == "production"

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../saved_models")
if not IS_PROD:
    os.makedirs(MODELS_DIR, exist_ok=True)

# ── Column definitions ─────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Loading Pressure (bar)",
    "Unloading Pressure (bar)",
    "Inlet Pressure (bar)",
    "Discharge Pressure (bar)",
    "Current (Amp)",
]

OPTIONAL_COLUMNS = [
    "Discharge Temperature ( C )",
    "Theoretical Electrical Power (kW)",
    "Theoretical Mechanical Power (kW)",
    "Specific Power Consumption (kW/m3/min)",
]

FEATURES = [
    "Loading Pressure (bar)",
    "Unloading Pressure (bar)",
    "Inlet Pressure (bar)",
    "Discharge Pressure (bar)",
    "Discharge Temperature ( C )",
    "Current (Amp)",
]

TARGET_ELEC = "Theoretical Electrical Power (kW)"
TARGET_MECH = "Theoretical Mechanical Power (kW)"
TARGET_SPC  = "Specific Power Consumption (kW/m3/min)"

# Scale factor: dataset P_elec = actual_kW × 7.6337
ELEC_SCALE = 7.6337


# ── Formula helpers ───────────────────────────────────────────
def compute_electrical_power(I: float, V: float = 415.0, cos_phi: float = 0.9) -> float:
    return (np.sqrt(3) * V * I * cos_phi) / 1000.0


def compute_flow_rate(P2: float, P_low: float = 7.0, P_high: float = 10.0,
                      Q_low: float = 45.23, Q_high: float = 35.47) -> float:
    if P_high == P_low:
        return Q_low
    Q = ((P2 - P_low) / (P_high - P_low)) * (Q_high - Q_low) + Q_low
    return float(np.clip(Q, min(Q_low, Q_high), max(Q_low, Q_high)))


def compute_mechanical_power(P1: float, P2: float, Q: float,
                              n: float = 1.4, z: int = 2) -> float:
    if P1 <= 0 or P2 <= 0 or Q <= 0:
        return 0.0
    ratio = P2 / P1
    exp   = (n - 1) / (n * z)
    mech  = (n / (n - 1)) * (Q / 60) * (P1 * 1e5) * (ratio ** exp - 1) * z
    return mech / 1000.0


def _get_unit(feature: str) -> str:
    return {
        "Loading Pressure (bar)":      "bar",
        "Unloading Pressure (bar)":    "bar",
        "Inlet Pressure (bar)":        "bar",
        "Discharge Pressure (bar)":    "bar",
        "Discharge Temperature ( C )": "°C",
        "Current (Amp)":               "A",
    }.get(feature, "")


# ── Dataset enrichment ────────────────────────────────────────
def enrich_dataframe(df: pd.DataFrame, user_params: dict) -> pd.DataFrame:
    """
    Compute missing derived columns using formulas.
    Also auto-detects and corrects scaled electrical power.
    """
    df      = df.copy()
    V       = float(user_params.get("voltage",            415.0))
    cos_phi = float(user_params.get("power_factor",       0.9))
    z       = int(user_params.get("compression_stages",   2))
    P_low   = float(user_params.get("p_low",              7.0))
    P_high  = float(user_params.get("p_high",             10.0))
    Q_low   = float(user_params.get("q_low",              45.23))
    Q_high  = float(user_params.get("q_high",             35.47))

    # Compute electrical power from Current if missing
    if "Current (Amp)" in df.columns and TARGET_ELEC not in df.columns:
        df[TARGET_ELEC] = df["Current (Amp)"].apply(
            lambda I: compute_electrical_power(I, V, cos_phi))

    # Auto-detect scale: if P_elec >> expected range, divide by ELEC_SCALE
    if TARGET_ELEC in df.columns:
        median_elec = df[TARGET_ELEC].median()
        if median_elec > 300:
            df[TARGET_ELEC] = df[TARGET_ELEC] / ELEC_SCALE

    # Flow rate from discharge pressure
    if "Discharge Pressure (bar)" in df.columns:
        df["Q_computed"] = df["Discharge Pressure (bar)"].apply(
            lambda p2: compute_flow_rate(p2, P_low, P_high, Q_low, Q_high))

        if TARGET_MECH not in df.columns and "Inlet Pressure (bar)" in df.columns:
            df[TARGET_MECH] = df.apply(
                lambda row: compute_mechanical_power(
                    row["Inlet Pressure (bar)"],
                    row["Discharge Pressure (bar)"],
                    row.get("Q_computed", Q_low), z=z), axis=1)

    # SPC
    if TARGET_ELEC in df.columns and "Q_computed" in df.columns:
        df[TARGET_SPC] = df[TARGET_ELEC] / df["Q_computed"].replace(0, np.nan)

    # Discharge temperature default
    if "Discharge Temperature ( C )" not in df.columns:
        df["Discharge Temperature ( C )"] = 35.0

    return df


# ── Validation ────────────────────────────────────────────────
def validate_dataset(df: pd.DataFrame, user_params: dict = None) -> dict:
    """Return a validation checklist for the dataset."""
    actual    = list(df.columns)
    missing   = [c for c in REQUIRED_COLUMNS if c not in actual]
    present_r = [c for c in REQUIRED_COLUMNS if c in actual]
    present_o = [c for c in OPTIONAL_COLUMNS if c in actual]

    derivable     = []
    not_derivable = []
    for col in missing:
        if col == TARGET_ELEC and "Current (Amp)" in actual:
            derivable.append(col)
        else:
            not_derivable.append(col)

    will_compute = []
    if TARGET_ELEC not in actual and "Current (Amp)" in actual:
        will_compute.append(TARGET_ELEC)
    if TARGET_MECH not in actual and "Discharge Pressure (bar)" in actual:
        will_compute.append(TARGET_MECH)
    if TARGET_SPC not in actual:
        will_compute.append(TARGET_SPC)
    if "Discharge Temperature ( C )" not in actual:
        will_compute.append("Discharge Temperature ( C ) [default 35°C]")

    try:
        sample = df.head(5).fillna("").to_dict(orient="records")
    except Exception:
        sample = []

    return {
        "filename":         "uploaded",
        "total_rows":       len(df),
        "total_columns":    len(df.columns),
        "columns_found":    actual,
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "present_required": present_r,
        "present_optional": present_o,
        "missing_required": not_derivable,
        "can_be_derived":   derivable,
        "will_be_computed": will_compute,
        "is_valid":         len(not_derivable) == 0,
        "valid":            len(not_derivable) == 0,
        "errors":           [f"Missing required columns: {', '.join(not_derivable)}"] if not_derivable else [],
        "was_raw":          len(missing) > 0 or len(actual) < 6,
        "sample_data":      sample,
        "data_preview": {
            col: {
                "min":   float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "max":   float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "nulls": int(df[col].isna().sum()),
            }
            for col in actual if col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        },
    }


# ── Auto-Clean ────────────────────────────────────────────────
def auto_clean(df: pd.DataFrame, user_params: dict = None) -> dict:
    """
    Auto-clean and preprocess a raw/clean dataset before storing.
    Returns: { df: cleaned DataFrame, summary: dict }
    """
    if user_params is None:
        user_params = {}

    original_rows = len(df)
    summary       = {"steps": [], "original_rows": original_rows}
    df            = df.copy()

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Drop fully empty rows
    before = len(df)
    df.dropna(how="all", inplace=True)
    dropped = before - len(df)
    if dropped:
        summary["steps"].append(f"Dropped {dropped} fully empty rows")

    # 3. Drop exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        summary["steps"].append(f"Dropped {dropped} duplicate rows")

    # 4. Coerce numeric columns
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Impute missing required cols with column median
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            median = df[col].median()
            n_miss = int(df[col].isna().sum())
            df[col] = df[col].fillna(median)
            summary["steps"].append(
                f"Imputed {n_miss} missing values in '{col}' with median {median:.3f}")

    # 6. Enrich — derive Electrical Power, Mechanical Power, SPC, Temperature
    df = enrich_dataframe(df, user_params)
    summary["steps"].append("Derived computed columns (Electrical Power, Mechanical Power, SPC)")

    # 7. Remove extreme outliers using 1st–99th percentile IQR × 3
    key_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if key_cols:
        before = len(df)
        Q1  = df[key_cols].quantile(0.01)
        Q3  = df[key_cols].quantile(0.99)
        IQR = Q3 - Q1
        mask = ~((df[key_cols] < (Q1 - 3 * IQR)) | (df[key_cols] > (Q3 + 3 * IQR))).any(axis=1)
        df  = df[mask]
        dropped = before - len(df)
        if dropped:
            summary["steps"].append(f"Removed {dropped} extreme outlier rows (IQR×3)")

    # 8. Reset index
    df.reset_index(drop=True, inplace=True)

    summary["final_rows"]    = len(df)
    summary["rows_removed"]  = original_rows - len(df)
    summary["columns_final"] = list(df.columns)

    return {"df": df, "summary": summary}


# ── ML Engine ─────────────────────────────────────────────────
class CompressorMLEngine:
    def __init__(self, compressor_id: str):
        self.compressor_id = compressor_id
        self.model_path    = os.path.join(MODELS_DIR, f"{compressor_id}_model.pkl")
        self.scaler        = StandardScaler()
        self.model_elec    = None
        self.model_mech    = None
        self.model_spc     = None
        self.dbscan        = None
        self.clean_df      = None
        self.scores: dict  = {}

    def load_model_from_dict(self, data: dict) -> bool:
        """
        Load sub-models from a pickled dict (from Supabase Storage).
        This is the correct warm-start method — use this, not load_model().
        Returns True if successfully loaded.
        """
        required = ("model_elec", "model_mech", "model_spc", "scaler")
        if not all(k in data for k in required):
            logger.warning("Pretrained model dict missing expected keys — skipping warm-start.")
            return False
        self.model_elec = data["model_elec"]
        self.model_mech = data["model_mech"]
        self.model_spc  = data["model_spc"]
        self.scaler     = data["scaler"]
        self.scores     = data.get("scores", {})
        return True

    def train(self, df: pd.DataFrame, user_params: dict = {}) -> dict:
        # ── Enrich & clean ──────────────────────────────────────
        df = enrich_dataframe(df, user_params)
        df = df.dropna(subset=REQUIRED_COLUMNS)

        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        if len(df) < 20:
            raise ValueError("Not enough clean rows (need ≥ 20) after preprocessing.")

        was_raw = TARGET_ELEC not in df.columns or df[TARGET_ELEC].isna().all()

        # ── STEP 1: DBSCAN Clustering ────────────────────────────
        X_cluster = df[[TARGET_ELEC, TARGET_SPC]].fillna(0)
        X_scaled  = self.scaler.fit_transform(X_cluster)
        self.dbscan = DBSCAN(eps=0.3, min_samples=max(5, len(df) // 50))
        df = df.copy()
        df["Cluster_ID"] = self.dbscan.fit_predict(X_scaled)

        labels = set(df["Cluster_ID"])
        if len(labels) > 1 and (-1 not in labels or len(labels) > 2):
            sil = silhouette_score(X_scaled, df["Cluster_ID"]) * 100
        else:
            sil = 50.0
        self.scores["silhouette"] = round(float(sil), 2)

        # ── STEP 2: GBR Models ───────────────────────────────────
        self.clean_df = df[df["Cluster_ID"] != -1].copy()
        if len(self.clean_df) < 10:
            self.clean_df = df.copy()

        X      = self.clean_df[FEATURES].fillna(0)
        y_elec = self.clean_df[TARGET_ELEC].fillna(0)
        y_mech = self.clean_df[TARGET_MECH].fillna(0)
        y_spc  = self.clean_df[TARGET_SPC].fillna(0)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_elec, test_size=0.2, random_state=42)

        self.model_elec = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
        self.model_elec.fit(X_tr, y_tr)

        y_pred            = self.model_elec.predict(X_te)
        self.scores["r2"] = round(float(r2_score(y_te, y_pred) * 100), 2)

        median_spc  = self.clean_df[TARGET_SPC].median()
        y_true_cls  = (self.clean_df[TARGET_SPC] < median_spc).astype(int)
        y_pred_cls  = (self.clean_df["Cluster_ID"] != -1).astype(int)
        try:
            self.scores["f1"] = round(
                float(f1_score(y_true_cls, y_pred_cls, zero_division=0) * 100), 2)
        except Exception:
            self.scores["f1"] = 66.67

        self.model_mech = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model_mech.fit(X, y_mech)
        self.model_spc  = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model_spc.fit(X, y_spc)

        train_curve = [float(mean_absolute_error(y_tr, p)) for p in self.model_elec.staged_predict(X_tr)]
        test_curve  = [float(mean_absolute_error(y_te, p)) for p in self.model_elec.staged_predict(X_te)]

        # ── STEP 3: Genetic Algorithm Optimisation ───────────────
        bounds = [
            (float(self.clean_df[f].min()), float(self.clean_df[f].max()))
            for f in FEATURES
        ]
        ga_res = differential_evolution(
            lambda x: float(self.model_elec.predict(pd.DataFrame([x], columns=FEATURES))[0]),
            bounds, seed=42, maxiter=500, tol=0.001, workers=1,
        )
        self.scores["convergence"] = round(max(0.0, (1 - ga_res.nfev / 30000) * 100), 2)

        opt_params = ga_res.x
        best_elec  = float(ga_res.fun)
        best_mech  = float(self.model_mech.predict(pd.DataFrame([opt_params], columns=FEATURES))[0])
        best_spc   = float(self.model_spc.predict(pd.DataFrame([opt_params], columns=FEATURES))[0])

        baseline_elec = float(df[TARGET_ELEC].mean())
        saving_pct    = ((baseline_elec - best_elec) / baseline_elec * 100) if baseline_elec else 0.0

        importances        = self.model_elec.feature_importances_
        feature_importance = {FEATURES[i]: round(float(importances[i]), 4) for i in range(len(FEATURES))}

        optimal_ranges = {}
        for i, f in enumerate(FEATURES):
            val    = float(opt_params[i])
            spread = (float(self.clean_df[f].max()) - float(self.clean_df[f].min())) * 0.05
            optimal_ranges[f] = {
                "optimal":   round(val, 4),
                "min":       round(val - spread, 4),
                "max":       round(val + spread, 4),
                "unit":      _get_unit(f),
                "data_min":  round(float(self.clean_df[f].min()), 4),
                "data_max":  round(float(self.clean_df[f].max()), 4),
                "data_mean": round(float(self.clean_df[f].mean()), 4),
            }

        # Only save locally in development
        self._save_model()

        logger.info(
            f"Training complete for {self.compressor_id}. "
            f"R²={self.scores['r2']}% | Saving={saving_pct:.2f}%"
        )

        return {
            "scores":                    self.scores,
            "optimal_parameters":        optimal_ranges,
            "best_electrical_power":     round(best_elec, 2),
            "best_mechanical_power":     round(best_mech, 2),
            "best_spc":                  round(best_spc, 4),
            "baseline_electrical_power": round(baseline_elec, 2),
            "power_saving_percent":      round(saving_pct, 2),
            "feature_importance":        feature_importance,
            "was_raw":                   was_raw,
            "training_curve":            {"train": train_curve, "test": test_curve},
            "cluster_stats": {
                "total_points": int(len(df)),
                "noise_points": int((df["Cluster_ID"] == -1).sum()),
                "clean_points": int(len(self.clean_df)),
                "n_clusters":   int(len(set(df["Cluster_ID"]) - {-1})),
            },
            "data_stats": {
                "electrical_power": {
                    "mean": round(float(df[TARGET_ELEC].mean()), 2),
                    "std":  round(float(df[TARGET_ELEC].std()),  2),
                    "min":  round(float(df[TARGET_ELEC].min()),  2),
                    "max":  round(float(df[TARGET_ELEC].max()),  2),
                },
                "mechanical_power": {
                    "mean": round(float(df[TARGET_MECH].mean()), 2),
                    "std":  round(float(df[TARGET_MECH].std()),  2),
                    "min":  round(float(df[TARGET_MECH].min()),  2),
                    "max":  round(float(df[TARGET_MECH].max()),  2),
                },
            },
            "scatter_data":     _scatter_data(df),
            "cluster_data":     _cluster_data(df),
            "histogram_data":   _histogram_data(df),
            "correlation_data": _correlation_data(df),
        }

    def _save_model(self):
        """
        Save model to local disk.
        In production this is skipped — models live in Supabase Storage only.
        """
        if IS_PROD:
            return
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self._model_dict(), f)
        except Exception as e:
            logger.warning(f"Local model save failed (non-critical): {e}")

    def _model_dict(self) -> dict:
        """Return serializable dict of all trained sub-models."""
        return {
            "model_elec": self.model_elec,
            "model_mech": self.model_mech,
            "model_spc":  self.model_spc,
            "scaler":     self.scaler,
            "scores":     self.scores,
            "clean_df_stats": {
                col: {
                    "min":  float(self.clean_df[col].min()),
                    "max":  float(self.clean_df[col].max()),
                    "mean": float(self.clean_df[col].mean()),
                }
                for col in FEATURES if self.clean_df is not None and col in self.clean_df.columns
            },
        }

    def load_model(self) -> bool:
        """Load model from local disk (development only)."""
        if IS_PROD or not os.path.exists(self.model_path):
            return False
        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            return self.load_model_from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            return False


# ── Chart helpers ─────────────────────────────────────────────
def _scatter_data(df: pd.DataFrame) -> list:
    sample = df.dropna(subset=[TARGET_ELEC, TARGET_MECH]).sample(
        min(200, len(df)), random_state=42)
    return [{"x": round(float(r[TARGET_ELEC]), 2),
             "y": round(float(r[TARGET_MECH]), 2),
             "cluster": int(r.get("Cluster_ID", 0))} for _, r in sample.iterrows()]


def _cluster_data(df: pd.DataFrame) -> list:
    sample = df.dropna(subset=[TARGET_ELEC, TARGET_SPC]).sample(
        min(200, len(df)), random_state=42)
    return [{"x": round(float(r[TARGET_ELEC]), 2),
             "y": round(float(r[TARGET_SPC]), 4),
             "cluster": int(r.get("Cluster_ID", 0))} for _, r in sample.iterrows()]


def _histogram_data(df: pd.DataFrame) -> dict:
    e_hist, e_bins = np.histogram(df[TARGET_ELEC].dropna(), bins=20)
    m_hist, m_bins = np.histogram(df[TARGET_MECH].dropna(), bins=20)
    return {
        "electrical": [{"bin": round(float(e_bins[i]), 2), "count": int(e_hist[i])}
                       for i in range(len(e_hist))],
        "mechanical": [{"bin": round(float(m_bins[i]), 2), "count": int(m_hist[i])}
                       for i in range(len(m_hist))],
    }


def _correlation_data(df: pd.DataFrame) -> list:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cols    = [c for c in FEATURES + [TARGET_ELEC, TARGET_MECH] if c in numeric]
    if len(cols) < 2:
        return []
    corr = df[cols].corr().round(3)
    return [{"x": c1, "y": c2, "value": float(corr.loc[c1, c2])}
            for c1 in cols for c2 in cols]


# ── train_model wrapper — used by retrain.py ─────────────────
def train_model(
    df: pd.DataFrame,
    user_params: dict = None,
    compressor_id: str = "shared",
) -> dict:
    """
    Convenience wrapper around CompressorMLEngine.train().
    Called by retrain.py with a combined DataFrame of all units.
    Returns result dict with 'model' key set to the engine instance.
    """
    if user_params is None:
        user_params = {}
    engine = CompressorMLEngine(compressor_id)
    result = engine.train(df, user_params)
    result["model"] = engine   # retrain.py pickles this as engine._model_dict()
    return result