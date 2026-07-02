"""
Train UTCI GPR Model — Version B
=================================
Εκπαιδεύει ένα Gaussian Process Regression μοντέλο (Rational Quadratic kernel)
για πρόγνωση UTCI, αναπαράγοντας το μοντέλο της διπλωματικής στη Python.

Το μοντέλο εκπαιδεύεται σε αντιπροσωπευτικό δείγμα 2000 σημείων — συνειδητή
επιλογή, καθώς το πλήρες GPR (14.809 σημεία) κλιμακώνει με O(n³) και γίνεται
απαγορευτικά αργό. Στα 2000 σημεία: R² ≈ 0.80, αρχείο ~47MB (κατάλληλο για
GitHub/Streamlit χωρίς Git LFS).

Το εκπαιδευμένο μοντέλο + ο scaler αποθηκεύονται στο 'utci_gpr_model.joblib',
το οποίο φορτώνει το live app.

------------------------------------------------------------------------
ΕΓΚΑΤΑΣΤΑΣΗ:
    pip install pandas numpy scikit-learn joblib

ΕΚΤΕΛΕΣΗ (μία φορά):
    python train_model.py
------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# 1. ΡΥΘΜΙΣΕΙΣ
# ============================================================
TRAINING_FILE = "TRAININGDATA_Regression.txt"
SAMPLE_SIZE = 2000          # μέγεθος δείγματος εκπαίδευσης
                            # (2000 → ~47MB αρχείο, R²≈0.80, γρήγορο deploy)
TEST_SIZE = 0.20            # 20% για test, όπως στη διπλωματική
RANDOM_STATE = 42           # για αναπαραγωγιμότητα
OUTPUT_MODEL = "utci_gpr_model.joblib"

# Η ΑΚΡΙΒΗΣ σειρά των 30 input στηλών — ΚΡΙΣΙΜΗ.
# Το live app πρέπει να κατασκευάζει τα inputs με ΑΚΡΙΒΩΣ αυτή τη σειρά.
FEATURE_COLUMNS = [
    "M4DB", "HR4DB", "T4DB", "RH4DB", "WS4DB", "GIRR4DB", "UTCI4DB",
    "M3DB", "HR3DB", "T3DB", "RH3DB", "WS3DB", "GIRR3DB", "UTCI3DB",
    "M2DB", "HR2DB", "T2DB", "RH2DB", "WS2DB", "GIRR2DB", "UTCI2DB",
    "M1DB", "HR1DB", "T1DB", "RH1DB", "WS1DB", "GIRR1DB", "UTCI1DB",
    "MND", "HRND",
]
TARGET_COLUMN = "UTCIND"


# ============================================================
# 2. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# ============================================================
print("Φόρτωση δεδομένων...")
data = pd.read_csv(TRAINING_FILE, sep="\t")
print(f"  Σύνολο: {data.shape[0]} παρατηρήσεις, {data.shape[1]} στήλες")

X = data[FEATURE_COLUMNS].values
y = data[TARGET_COLUMN].values


# ============================================================
# 3. ΔΙΑΧΩΡΙΣΜΟΣ TRAIN / TEST
# ============================================================
# Πρώτα κρατάμε σταθερό test set (20%) — δεν συμμετέχει ΠΟΤΕ στην εκπαίδευση.
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Από το υπόλοιπο 80%, παίρνουμε αντιπροσωπευτικό δείγμα για εκπαίδευση.
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_pool), size=min(SAMPLE_SIZE, len(X_pool)), replace=False)
X_train = X_pool[sample_idx]
y_train = y_pool[sample_idx]

print(f"  Training: {len(X_train)} σημεία (δείγμα)")
print(f"  Test:     {len(X_test)} σημεία (σταθερό)")


# ============================================================
# 4. ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ (SCALING)
# ============================================================
# Το GPR αποδίδει καλύτερα με κανονικοποιημένα inputs.
# ΣΗΜΑΝΤΙΚΟ: ο scaler "μαθαίνει" ΜΟΝΟ από τα training δεδομένα,
# και μετά εφαρμόζεται και στο test (και live) με τις ίδιες παραμέτρους.
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


# ============================================================
# 5. ΕΚΠΑΙΔΕΥΣΗ GPR
# ============================================================
print("\nΕκπαίδευση GPR (Rational Quadratic kernel)...")

# Rational Quadratic kernel — ο ίδιος που βγήκε νικητής στο MATLAB.
# ConstantKernel: ρυθμίζει τη συνολική κλίμακα.
# alpha=0.1: ρυθμίζει τον θόρυβο (regularization).
kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.1,
    normalize_y=True,
    n_restarts_optimizer=0,
)

t0 = time.time()
gpr.fit(X_train_s, y_train)
elapsed = time.time() - t0
print(f"  Ολοκληρώθηκε σε {elapsed:.1f} δευτερόλεπτα")
print(f"  Βελτιστοποιημένο kernel: {gpr.kernel_}")


# ============================================================
# 6. ΑΞΙΟΛΟΓΗΣΗ ΣΤΟ TEST SET
# ============================================================
y_pred = gpr.predict(X_test_s)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nΑποτελέσματα στο ανεξάρτητο test set:")
print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.2f} °C")


# ============================================================
# 7. ΑΠΟΘΗΚΕΥΣΗ ΜΟΝΤΕΛΟΥ + SCALER
# ============================================================
# Αποθηκεύουμε ΜΑΖΙ το μοντέλο, τον scaler, και τη σειρά των features,
# ώστε το live app να τα φορτώσει και να τα χρησιμοποιήσει σωστά.
bundle = {
    "model": gpr,
    "scaler": scaler,
    "feature_columns": FEATURE_COLUMNS,
    "metrics": {"r2": r2, "rmse": rmse, "n_train": len(X_train)},
}
joblib.dump(bundle, OUTPUT_MODEL)
print(f"\n✅ Μοντέλο αποθηκεύτηκε: {OUTPUT_MODEL}")
print("   Έτοιμο για χρήση στο live app.")
