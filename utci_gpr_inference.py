"""
UTCI GPR — Φάση 2: Inference από Open-Meteo
============================================
Ανακατασκευάζει τα 30 features που περιμένει το εκπαιδευμένο GPR μοντέλο
(utci_gpr_model.joblib) απευθείας από το Open-Meteo API, τρέχει την πρόβλεψη
με uncertainty band, και επιστρέφει και το "επίσημο" UTCI για σύγκριση.

Σχεδιαστική αρχή (ΚΡΙΣΙΜΗ): η ΙΔΙΑ compute_utci() χρησιμοποιείται
  (α) για τα 4 lag-UTCI που τρέφουν το GPR, και
  (β) για το official baseline της target ώρας.
Έτσι η σύγκριση GPR-vs-official είναι apples-to-apples και τα lag-UTCI είναι
συνεπή με τον υπολογισμό του Version A. ΒΑΛΕ ΕΔΩ το MRT/UTCI του Version A.

Εξαρτήσεις: requests, numpy, joblib, scikit-learn (ΙΔΙΑ version με το training!),
            pythermalcomfort
------------------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import requests

# ── Ρυθμίσεις ────────────────────────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Η ΑΚΡΙΒΗΣ σειρά — πρέπει να ταυτίζεται με το bundle["feature_columns"].
FEATURE_COLUMNS = [
    "M4DB", "HR4DB", "T4DB", "RH4DB", "WS4DB", "GIRR4DB", "UTCI4DB",
    "M3DB", "HR3DB", "T3DB", "RH3DB", "WS3DB", "GIRR3DB", "UTCI3DB",
    "M2DB", "HR2DB", "T2DB", "RH2DB", "WS2DB", "GIRR2DB", "UTCI2DB",
    "M1DB", "HR1DB", "T1DB", "RH1DB", "WS1DB", "GIRR1DB", "UTCI1DB",
    "MND", "HRND",
]

# Μεταβλητές που ζητάμε από το Open-Meteo, με τις ΣΩΣΤΕΣ μονάδες.
_HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"


# ── UTCI / MRT ───────────────────────────────────────────────────────────────
# ✅ Ταυτισμένο με το Version A (estimate_mrt + compute_utci_series):
#    σκιά → MRT=Tair, ήλιος → MRT=Tair+0.025·shortwave. Η ΙΔΙΑ compute_utci
#    τρέφει και τα lag-UTCI και το official baseline (apples-to-apples).

def compute_mrt(tdb: float, ghi: float, sun: bool) -> float:
    """Mean Radiant Temperature — ίδια λογική με το estimate_mrt του Version A."""
    if sun:
        return tdb + 0.025 * ghi   # στον ήλιο η ακτινοβολία ανεβάζει την MRT
    return tdb                      # στη σκιά, MRT ≈ θερμοκρασία αέρα


def compute_utci(tdb: float, rh: float, ws: float, ghi: float, sun: bool) -> float:
    """UTCI (°C) — ίδια λογική με το compute_utci_series του Version A.
    Χρησιμοποιείται και για τα lag-UTCI και για το official baseline."""
    from pythermalcomfort.models import utci as _utci
    tr = compute_mrt(tdb, ghi, sun)
    v = max(ws, 0.5)               # Version A: μόνο κάτω όριο (ο UTCI θέλει ≥0.5 m/s)
    result = _utci(tdb=tdb, tr=tr, v=v, rh=rh)
    return float(result.utci if hasattr(result, "utci") else result["utci"])


# ── Open-Meteo fetch ─────────────────────────────────────────────────────────
def _fetch_hourly(lat: float, lon: float, start: date, end: date) -> dict[str, dict]:
    """Επιστρέφει {iso_timestamp: {"T":..,"RH":..,"WS":..,"GIRR":..}} σε τοπική ώρα."""
    r = requests.get(OPEN_METEO_URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": _HOURLY_VARS,
        "wind_speed_unit": "ms",          # ⚠️ αλλιώς km/h → ×3.6 skew στο WS
        "timezone": "auto",               # ⚠️ τοπικές ώρες, όπως η διπλωματική
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    out = {}
    for i, t in enumerate(h["time"]):
        out[t] = {
            "T":    h["temperature_2m"][i],
            "RH":   h["relative_humidity_2m"][i],
            "WS":   h["wind_speed_10m"][i],
            "GIRR": h["shortwave_radiation"][i],
        }
    return out


# ── Ανακατασκευή του feature vector ──────────────────────────────────────────
@dataclass
class Reconstruction:
    features: dict[str, float]          # named, για debugging/logging
    vector: np.ndarray                  # (30,) στη σειρά FEATURE_COLUMNS
    official_utci: float                # UTCI στην target ώρα από Open-Meteo forecast


def build_features(lat: float, lon: float, target: datetime, sun: bool) -> Reconstruction:
    """Χτίζει τα 30 inputs για πρόβλεψη UTCI στη target ώρα, από τις 4 προηγούμενες
    ημερολογιακές μέρες στην ΙΔΙΑ ώρα. `target` = μέρα+ώρα πρόβλεψης (τοπική)."""
    H = target.hour
    lag_days = [target.date() - timedelta(days=k) for k in (4, 3, 2, 1)]  # 4DB→1DB

    win = _fetch_hourly(lat, lon, min(lag_days), target.date())

    def at(d: date) -> dict:
        key = f"{d.isoformat()}T{H:02d}:00"
        if key not in win:
            raise KeyError(f"Λείπει timestamp {key} από το Open-Meteo (gap;)")
        return win[key]

    feats: dict[str, float] = {}
    for lag, d in zip(("4DB", "3DB", "2DB", "1DB"), lag_days):
        m = at(d)
        utci_lag = round(compute_utci(m["T"], m["RH"], m["WS"], m["GIRR"], sun))
        feats[f"M{lag}"]    = d.month
        feats[f"HR{lag}"]   = H
        feats[f"T{lag}"]    = m["T"]
        feats[f"RH{lag}"]   = m["RH"]
        feats[f"WS{lag}"]   = m["WS"]
        feats[f"GIRR{lag}"] = m["GIRR"]
        feats[f"UTCI{lag}"] = utci_lag
    feats["MND"]  = target.month
    feats["HRND"] = H

    # Official UTCI στην ΙΔΙΑ target ώρα (Open-Meteo forecast) — για σύγκριση.
    tgt = at(target.date())
    official = compute_utci(tgt["T"], tgt["RH"], tgt["WS"], tgt["GIRR"], sun)

    vector = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=float)
    return Reconstruction(features=feats, vector=vector, official_utci=official)


# ── Πρόβλεψη ─────────────────────────────────────────────────────────────────
def predict(bundle: dict, recon: Reconstruction) -> dict:
    """GPR πρόβλεψη με uncertainty. Το bundle είναι το joblib.load(...)."""
    assert bundle["feature_columns"] == FEATURE_COLUMNS, \
        "Ασυμφωνία σειράς features με το εκπαιδευμένο μοντέλο!"
    Xs = bundle["scaler"].transform(recon.vector.reshape(1, -1))
    mean, std = bundle["model"].predict(Xs, return_std=True)
    gpr = float(mean[0])
    return {
        "gpr_utci": gpr,
        "gpr_std": float(std[0]),          # ±1σ confidence band
        "official_utci": recon.official_utci,
        "delta": gpr - recon.official_utci,
    }


# ── Self-test (offline, χωρίς δίκτυο) ────────────────────────────────────────
if __name__ == "__main__":
    import joblib, warnings
    warnings.filterwarnings("ignore")
    b = joblib.load("utci_gpr_model.joblib")

    # Ταΐζουμε μια πραγματική γραμμή training ως vector για να επαληθεύσουμε
    # ότι scaler.transform + model.predict δουλεύουν στο ΙΔΙΟ path.
    demo = np.array([4,23,18.06,41.93,4,0,6, 4,23,17.94,54.97,0.4,0,16,
                     4,23,19.17,54.93,0.4,0,17, 4,23,19.28,56.94,0.4,0,17,
                     4,23], dtype=float)
    r = Reconstruction(features={}, vector=demo, official_utci=float("nan"))
    Xs = b["scaler"].transform(r.vector.reshape(1, -1))
    mean, std = b["model"].predict(Xs, return_std=True)
    print(f"Self-test row → GPR={mean[0]:.1f}°C ±{std[0]:.2f}  (actual UTCIND=17)")
    print("Inference path OK. Για live: predict(b, build_features(lat, lon, target, sun)).")
