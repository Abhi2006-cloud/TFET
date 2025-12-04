# TFET – Line-TFET ML Predictor

Machine‑learning workflow to model the \(I_D\)–\(V_G\) characteristics of a Line‑TFET device:

- **`train_pipeline.py`**: generates synthetic (or real) TFET data, does feature engineering, trains a Random Forest on \(\log(I_D)\), and saves artifacts.
- **`api_fastapi.py`**: FastAPI service that exposes a `/predict_curve` endpoint.
- **`streamlit_app.py`**: interactive Streamlit UI to choose device parameters and visualize the predicted \(I_D\)–\(V_G\) curve.

The GitHub repo is here: `https://github.com/Abhi2006-cloud/TFET`.

---

## 1. Setup

From the project directory:

```bash
cd /Users/abhisuryawanshi/Desktop/project-Y

# (optional) create venv if it doesn't exist
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib joblib streamlit fastapi uvicorn
```

> The `venv/` and `saved/` directories are ignored by git; they are local only.

---

## 2. Train the model

This step creates the model and scaler artifacts under `saved/` and also updates the convenient `tfet_latest_*.pkl` files that the apps use.

Use synthetic data (quick start):

```bash
source venv/bin/activate
python train_pipeline.py --samples 10000 --n_iter 10
```

You should see logs ending with R² scores and a message like:

> `[TRAIN] Saved artifacts to saved/tfet_model_YYYYMMDD_HHMMSS_* and updated tfet_latest_*`

If you have **real long‑format CSV data**, place it somewhere and run:

```bash
python train_pipeline.py --data /path/to/your_data.csv --n_iter 30
```

The CSV must contain columns:

- `device_id`, `Lov`, `tn`, `tox`, `WK`, `VG`, `ID`

---

## 3. Run the FastAPI service

Start the API server:

```bash
source venv/bin/activate
uvicorn api_fastapi:app --reload --port 8000
```

- Open `http://localhost:8000/` to see a simple JSON message.
- Open interactive docs at `http://localhost:8000/docs`.

### 3.1 Predict curve via API

Send a POST request to `/predict_curve`:

```bash
curl -X POST "http://localhost:8000/predict_curve" \
  -H "Content-Type: application/json" \
  -d '{
        "Lov": 4.0,
        "tn": 2.0,
        "tox": 2.5,
        "WK": 4.25,
        "VG_start": 0.0,
        "VG_end": 0.6,
        "points": 200
      }'
```

Response structure:

```json
{
  "VG": [ ... ],
  "ID": [ ... ]
}
```

Where `VG` and `ID` are lists of equal length.

---

## 4. Run the Streamlit UI

Launch the app:

```bash
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

Then open:

- `http://localhost:8501` in your browser.

### 4.1 Using the UI

In the **sidebar**:

- Adjust device parameters:
  - `Lov` (nm)
  - `tn` (nm)
  - `tox` (nm)
  - `WK` (eV)
  - Number of `VG points`
- Click **“Predict ID–VG Curve”**:
  - The app sweeps `VG` from 0.0 to 0.6 V.
  - It applies the same feature engineering and scaling used during training.
  - The Random Forest predicts \(\log(I_D)\), which is exponentiated to get \(I_D\).
  - A **semilog plot** of \(I_D\) vs. \(V_G\) is shown.
  - A small preview table and a **“Download CSV”** button are provided.

You can also click **“Show Feature Importances”** to see the learned importance of the engineered features.

---

## 5. Internals / Structure

- `tfet_feature_engineer.py`
  - Contains the `TFETFeatureEngineer` transformer used by both training and inference.
  - Also includes `register_pickle_aliases(...)`, which makes older joblib artifacts (that refer to `main.TFETFeatureEngineer`) loadable.

- `train_pipeline.py`
  - Generates synthetic data (`generate_synthetic_tfet_data_improved`).
  - Applies physics‑inspired feature engineering.
  - Splits data (stratified by `VG` quantiles).
  - Runs `RandomizedSearchCV` on `RandomForestRegressor`.
  - Saves:
    - Timestamped artifacts: `saved/tfet_model_<timestamp>_{rf,scaler,fe}.pkl`
    - Latest pointers: `saved/tfet_latest_{rf,scaler,fe}.pkl`

- `api_fastapi.py`
  - Lazily loads the latest artifacts on first request.
  - Exposes `POST /predict_curve`.

- `streamlit_app.py`
  - Loads the latest artifacts on import.
  - Provides sliders + plotting + CSV download.

---

## 6. Git / GitHub

This repo is already set up with:

- Default branch: `main`
- Remote: `origin` → `https://github.com/Abhi2006-cloud/TFET.git`

Typical workflow:

```bash
# after making changes
git status
git add <files>
git commit -m "Describe your change"
git push
```

---

## 7. Troubleshooting

- **“Error loading model artifacts … Run training first.”**
  - Run `python train_pipeline.py` after activating the venv.
  - Confirm that `saved/tfet_latest_rf.pkl`, `saved/tfet_latest_scaler.pkl`, and `saved/tfet_latest_fe.pkl` exist.

- **Streamlit app doesn’t load or crashes at start**
  - Ensure you are in the venv: `source venv/bin/activate`.
  - Reinstall dependencies if needed:
    ```bash
    pip install --upgrade pip
    pip install numpy pandas scikit-learn matplotlib joblib streamlit fastapi uvicorn
    ```

- **API returns weird values**
  - Re‑train the model with more samples or different noise / hyperparameters:
    ```bash
    python train_pipeline.py --samples 30000 --n_iter 30
    ```

If you have any particular use case (e.g., plugging in your own measured data), start from `train_pipeline.py` and replace the synthetic data with your CSV via the `--data` option.


