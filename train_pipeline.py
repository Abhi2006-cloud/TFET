#!/usr/bin/env python3
"""
train_pipeline.py
- Generates synthetic TFET data (or loads long-format CSV if provided)
- Feature engineering (physics-inspired)
- Trains RandomForest on log(ID)
- RandomizedSearchCV hyperparam tuning
- Saves timestamped artifacts and a tfet_latest_* copy for serving
Usage:
    python train_pipeline.py              # uses synthetic data (quick)
    python train_pipeline.py --data path  # to use your real long-format CSV
"""

import os, time, argparse, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

from tfet_feature_engineer import TFETFeatureEngineer
warnings.filterwarnings("ignore")

os.makedirs("saved", exist_ok=True)

# ---------------- Synthetic data generator (physics-inspired) ----------------
def generate_synthetic_tfet_data_improved(n_samples=20000, noise_scale=5e-2, random_state=42):
    rng = np.random.RandomState(random_state)
    Lov = rng.uniform(0.1, 10.0, n_samples)    # nm
    tn  = rng.uniform(1.0, 5.0, n_samples)     # nm
    tox = rng.uniform(1.0, 6.0, n_samples)     # nm
    WK  = rng.uniform(4.1, 4.5, n_samples)     # eV
    VG = rng.uniform(0.0, 0.6, n_samples)      # V

    ID_clean = (
        1e-12 * np.exp( 6.0 * (VG - 0.25) )
        * (Lov ** 0.85)
        * np.exp(-0.45 * tn)
        * np.exp(-0.28 * (tox - 1.0))
        * np.exp(-2.3 * (WK - 4.2))
    )
    noise = 1.0 + rng.normal(0.0, noise_scale, n_samples)
    ID = ID_clean * noise
    ID = np.clip(ID, 1e-25, None)

    df = pd.DataFrame({"Lov": Lov, "tn": tn, "tox": tox, "WK": WK, "VG": VG, "ID": ID})
    return df

# ---------------- Utility: load long-format CSV ----------------
def load_long_csv(path):
    df = pd.read_csv(path)
    required = {"device_id","Lov","tn","tox","WK","VG","ID"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")
    return df

# ---------------- Training pipeline ----------------
def train(df=None, use_synthetic=True, n_samples=20000, random_state=42, n_iter=30):
    # 1) data
    if df is None and use_synthetic:
        print("[TRAIN] Generating synthetic dataset...")
        df = generate_synthetic_tfet_data_improved(n_samples=n_samples, noise_scale=5e-2, random_state=random_state)
    elif df is None:
        raise ValueError("No dataframe provided and use_synthetic=False")

    print(f"[TRAIN] Data samples: {len(df)}")

    # 2) prepare X,y
    X_raw = df[["Lov","tn","tox","WK","VG"]]
    y = df["ID"].values
    y_log = np.log(np.clip(y, 1e-25, None))

    # 3) stratified-like split by VG quantiles
    vg_bins = pd.qcut(df["VG"], q=10, duplicates='drop', labels=False)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_log, test_size=0.2, random_state=random_state, stratify=vg_bins)

    # 4) feature eng + scale
    fe = TFETFeatureEngineer()
    scaler = StandardScaler()
    X_train_fe = fe.transform(X_train)
    X_test_fe = fe.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_fe)
    X_test_scaled = scaler.transform(X_test_fe)

    # 5) Randomized search for RandomForest
    print("[TRAIN] RandomizedSearchCV starting...")
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    param_dist = {
        'n_estimators': [200, 400, 600],
        'max_depth': [12, 18, 25, None],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', 0.6, 0.8]
    }
    rs = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, scoring='r2', cv=3, random_state=random_state, n_jobs=-1, verbose=1)
    rs.fit(X_train_scaled, y_train)

    print("[TRAIN] Best params:", rs.best_params_)
    best_rf = rs.best_estimator_
    best_rf.fit(X_train_scaled, y_train)

    # 6) Evaluate
    ylog_pred = best_rf.predict(X_test_scaled)
    y_pred = np.exp(ylog_pred)
    r2_log = r2_score(y_test, ylog_pred)
    r2_orig = r2_score(np.exp(y_test), y_pred)
    print(f"[TRAIN] R2 (log-domain): {r2_log:.4f}")
    print(f"[TRAIN] R2 (original): {r2_orig:.4f}")

    # 7) save artifacts (timestamped + latest)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"saved/tfet_model_{ts}"
    joblib.dump(best_rf, base + "_rf.pkl")
    joblib.dump(scaler, base + "_scaler.pkl")
    joblib.dump(fe, base + "_fe.pkl")
    pd.DataFrame({"r2_log":[r2_log],"r2_orig":[r2_orig]}).to_csv(base + "_metrics.csv", index=False)

    # create/update latest symlink (or copy for windows)
    latest_rf = "saved/tfet_latest_rf.pkl"
    latest_scaler = "saved/tfet_latest_scaler.pkl"
    latest_fe = "saved/tfet_latest_fe.pkl"
    joblib.dump(best_rf, latest_rf)
    joblib.dump(scaler, latest_scaler)
    joblib.dump(fe, latest_fe)

    print(f"[TRAIN] Saved artifacts to {base}_* and updated tfet_latest_*")

    return {"model": best_rf, "scaler": scaler, "fe": fe, "metrics": {"r2_log": r2_log, "r2_orig": r2_orig}}

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="path to long-format CSV (device_id,Lov,tn,tox,WK,VG,ID). If omitted, synthetic data used.")
    parser.add_argument("--samples", type=int, default=20000, help="synthetic samples if using synthetic")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=30, help="RandomizedSearchCV iterations")
    args = parser.parse_args()

    if args.data:
        print("[CLI] Loading CSV:", args.data)
        df_real = load_long_csv(args.data)
        train(df=df_real, use_synthetic=False, random_state=args.random_state, n_iter=args.n_iter)
    else:
        train(use_synthetic=True, n_samples=args.samples, random_state=args.random_state, n_iter=args.n_iter)
