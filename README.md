# UTCI Forecasting — from analytic index to a trained GP model

Two complementary Streamlit apps that estimate the **Universal Thermal Climate
Index (UTCI)** for any location, using live [Open-Meteo](https://open-meteo.com)
data. Together they show the progression from a physics-based dashboard to a
machine-learning forecaster that reproduces and deploys the model from my
diploma thesis.

| | What it does | Live |
|---|---|---|
| **Version A — Analytic dashboard** | Computes UTCI with the official formula (`pythermalcomfort`) over a 24h forecast, with a sun/shade MRT toggle and city search. | https://utci--dashboard.streamlit.app |
| **Version B — GPR forecaster** | Predicts UTCI with a Gaussian Process Regression model retrained from my thesis, then compares it against the analytic formula with a quantified uncertainty band. | *(this repo — set as the Streamlit main file)* |

---

## Version B — what it actually does

The thesis model forecasts UTCI at hour *H* of a target day from the **same hour
over the previous four days**: 7 variables (month, hour, air temperature,
relative humidity, wind speed, global irradiance, UTCI) × 4 lag days, plus the
target month and hour — 30 inputs in total, mapping to the target UTCI.

At inference time the app reconstructs those 30 inputs live from Open-Meteo
(`past_days`), computes the four lag-UTCI values with the *same* MRT/UTCI logic
as Version A (so the comparison is apples-to-apples), scales them with the
training-time `StandardScaler`, and runs the GPR.

- **Model:** Gaussian Process Regression, Rational Quadratic kernel
  (the kernel that won in the original MATLAB study), retrained in scikit-learn.
- **Training:** 2,000-point representative sample of 14,809 observations
  (full GPR scales O(n³); the sample keeps the artifact deployable without Git LFS).
- **Free advantage of GPR:** `predict(return_std=True)` yields a confidence band
  that neither the analytic formula nor the thesis ANN provides.

## Honest evaluation

- **R² = 0.80, RMSE = 4.25 °C** on a held-out test set.
- **Beats naive persistence** (predicting "same as yesterday"): persistence
  scores RMSE ≈ 5.77 °C, so the model cuts error by ~26 %. Temperature lags — not
  the UTCI lags — are the strongest predictors, so this is genuine multivariate
  regression, not autoregression.
- **No systematic hourly bias:** mean residual against the true target is
  < 0.03 °C across every hour of the day. This validation showed that the gap
  between the GPR line and the analytic line in the live app is largely
  **definitional** (station-based thesis UTCI vs. formula-on-forecast UTCI),
  not model error.

### Known limitation
The MRT estimate is a simplified radiation-based model, shared with Version A.
It is internally consistent, but it is not guaranteed to match the exact MRT
convention that produced the thesis training data — a bounded source of skew on
the UTCI-lag inputs, documented rather than hidden.

## Tech stack
Python · scikit-learn · pythermalcomfort · Streamlit · Open-Meteo API · Altair

## Run locally
```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash; use venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run gpr_forecast_section.py
```
Retrain the model (optional, one-off):
```bash
python train_model.py             # writes utci_gpr_model.joblib
```

## Repository layout
- `gpr_forecast_section.py` — the Streamlit app (Version B)
- `utci_gpr_inference.py` — Open-Meteo → 30-feature reconstruction + prediction
- `train_model.py` — training script (reproducible)
- `utci_gpr_model.joblib` — trained model + scaler + feature order
- `TRAININGDATA_Regression.txt`, `TARGETDATA.txt` — thesis dataset

## Data & attribution
Weather data © Open-Meteo.com (CC-BY 4.0). UTCI via `pythermalcomfort`.
Model and dataset derived from my diploma thesis (University of West Attica).
