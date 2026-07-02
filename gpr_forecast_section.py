"""
Streamlit section — Φάση 2: GPR Forecast vs Official UTCI
=========================================================
Drop-in κομμάτι για το Version A app. Καλείς render_gpr_forecast(lat, lon, sun)
με τα widgets που ΗΔΗ έχει το Version A (city search → lat/lon, sun/shade toggle).

Δύο τρόποι χρήσης:
  1) Integration: import render_gpr_forecast στο app.py σου και κάλεσέ το.
  2) Standalone test: `streamlit run gpr_forecast_section.py`
     (έχει minimal δικά του widgets στο κάτω μέρος).

Απαιτεί: το utci_gpr_inference.py και το utci_gpr_model.joblib στον ίδιο φάκελο,
+ ΙΔΙΑ scikit-learn version με το training (κάρφωσέ την στο requirements.txt).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from utci_gpr_inference import build_features, predict


# ── UTCI thermal-stress κατηγορίες (επίσημη κλίμακα) ─────────────────────────
_UTCI_SCALE = [
    (46,  float("inf"), "Ακραίο θερμικό stress",       "#7f0000"),
    (38,  46,           "Πολύ ισχυρό θερμικό stress",  "#b30000"),
    (32,  38,           "Ισχυρό θερμικό stress",       "#e34a33"),
    (26,  32,           "Μέτριο θερμικό stress",       "#fc8d59"),
    (9,   26,           "Χωρίς θερμικό stress",        "#2ca25f"),
    (0,   9,            "Ελαφρύ ψυχρό stress",         "#a6bddb"),
    (-13, 0,            "Μέτριο ψυχρό stress",         "#74a9cf"),
    (-27, -13,          "Ισχυρό ψυχρό stress",         "#3690c0"),
    (-40, -27,          "Πολύ ισχυρό ψυχρό stress",    "#0570b0"),
    (float("-inf"), -40,"Ακραίο ψυχρό stress",         "#034e7b"),
]


def utci_category(value: float) -> tuple[str, str]:
    """UTCI (°C) → (ετικέτα κατηγορίας, χρώμα)."""
    for lo, hi, label, color in _UTCI_SCALE:
        if lo <= value < hi:
            return label, color
    return "—", "#888888"


def _badge(label: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:10px;font-size:0.85em;white-space:nowrap'>{label}</span>")


# ── Cache: μοντέλο (μία φορά) + forecast (ttl) ───────────────────────────────
@st.cache_resource
def load_bundle(path: str = "utci_gpr_model.joblib"):
    import joblib
    return joblib.load(path)


@st.cache_data(ttl=900, show_spinner=False)  # 15' cache ανά (τοποθεσία, target, sun)
def _run_forecast(lat: float, lon: float, target_iso: str, sun: bool, _bundle) -> dict:
    target = datetime.fromisoformat(target_iso)
    recon = build_features(lat, lon, target, sun)
    out = predict(_bundle, recon)
    out["features"] = recon.features
    return out


@st.cache_data(ttl=86400, show_spinner=False)   # συντεταγμένες πόλεων ~ σταθερές
def geocode_city(name: str, count: int = 5) -> list[dict]:
    """Open-Meteo Geocoding: όνομα πόλης → λίστα υποψήφιων τοποθεσιών.
    ΙΔΙΟ geocoder με το Version A. Στο integration ΜΗΝ βάλεις δεύτερο πεδίο πόλης —
    ξαναχρησιμοποίησε το city search του Version A και πέρνα lat/lon στο render_gpr_forecast."""
    if not name.strip():
        return []
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": count, "language": "el"}, timeout=15)
    r.raise_for_status()
    return r.json().get("results", []) or []


# ── Το section ───────────────────────────────────────────────────────────────
def render_gpr_forecast(lat: float, lon: float, sun: bool) -> None:
    """Ζωγραφίζει το GPR-vs-official forecast για δεδομένα lat/lon/sun."""
    st.subheader("🔮 Πρόβλεψη UTCI με μοντέλο GPR")
    st.caption("Πρόβλεψη UTCI από τις 4 προηγούμενες μέρες στην ίδια ώρα — "
               "αναπαραγωγή του GPR μοντέλου της διπλωματικής, με uncertainty band.")

    # Target: μέρα + ώρα πρόβλεψης (default: αύριο, τρέχουσα ώρα).
    c1, c2 = st.columns(2)
    with c1:
        tgt_date = st.date_input("Μέρα πρόβλεψης",
                                 value=datetime.now().date() + timedelta(days=1))
    with c2:
        tgt_hour = st.selectbox("Ώρα (τοπική)", list(range(24)),
                                index=min(datetime.now().hour, 23))
    target = datetime.combine(tgt_date, datetime.min.time()).replace(hour=int(tgt_hour))

    bundle = load_bundle()
    try:
        with st.spinner("Ανάκτηση Open-Meteo & πρόβλεψη…"):
            out = _run_forecast(lat, lon, target.isoformat(), sun, bundle)
    except KeyError as e:
        st.warning(f"Το Open-Meteo δεν επέστρεψε όλες τις ώρες που χρειάζονται "
                   f"(πιθανό gap): {e}. Δοκίμασε άλλη μέρα/ώρα.")
        return
    except Exception as e:
        st.error(f"Αποτυχία πρόβλεψης: {e}")
        return

    gpr, std = out["gpr_utci"], out["gpr_std"]
    official, delta = out["official_utci"], out["delta"]
    g_label, g_color = utci_category(gpr)
    o_label, o_color = utci_category(official)

    # Νυχτερινό hint: αν αυτή την ώρα η ακτινοβολία≈0, το sun toggle είναι αδρανές.
    if sun and out["features"].get("GIRR1DB", 0) < 10:
        st.caption("🌙 Αυτή την ώρα ο ήλιος έχει δύσει (ακτινοβολία≈0) — "
                   "το «Ήλιος» δεν επηρεάζει· MRT = θερμοκρασία αέρα.")

    # Τρία metrics — το delta είναι ΟΥΔΕΤΕΡΟ (διαφορά μεθόδου, όχι σφάλμα).
    m1, m2, m3 = st.columns(3)
    m1.metric("GPR πρόβλεψη", f"{gpr:.1f} °C",
              f"{delta:+.1f} °C έναντι τύπου", delta_color="off",
              help="Μοντέλο εκπαιδευμένο στα δεδομένα της διπλωματικής (station-based UTCI).")
    m2.metric("Αναλυτικός τύπος", f"{official:.1f} °C",
              help="pythermalcomfort στην target ώρα, πάνω στο Open-Meteo forecast.")
    m3.metric("Αβεβαιότητα (±1σ)", f"± {std:.1f} °C",
              help="Διαθέσιμο μόνο στο GPR — ούτε ο αναλυτικός τύπος ούτε το ANN το δίνουν.")

    st.caption(
        "ℹ️ Οι δύο τιμές προέρχονται από **διαφορετικές μεθόδους** — μοντέλο σε "
        "δεδομένα σταθμού vs αναλυτικός τύπος πάνω στην πρόγνωση Open-Meteo. Η "
        "διαφορά τους είναι εν μέρει **ορισμού**, όχι σφάλμα: το GPR δεν έχει "
        "μετρήσιμη συστηματική κλίση έναντι του UTCI της διπλωματικής "
        "(mean residual < 0.03°C σε όλες τις ώρες)."
    )

    b1, b2 = st.columns(2)
    b1.markdown("GPR → " + _badge(g_label, g_color), unsafe_allow_html=True)
    b2.markdown("Τύπος → " + _badge(o_label, o_color), unsafe_allow_html=True)

    # Οπτικοποίηση: GPR point + error band vs official, πάνω σε UTCI άξονα.
    lo, hi = min(gpr - std, official) - 3, max(gpr + std, official) + 3
    band = pd.DataFrame({"x": [gpr - std, gpr + std], "y": ["GPR", "GPR"]})
    pts = pd.DataFrame({
        "UTCI": [gpr, official],
        "Μοντέλο": ["GPR", "Τύπος"],
        "χρώμα": [g_color, o_color],
    })
    try:
        import altair as alt
        rule = alt.Chart(band).mark_line(size=10, opacity=0.25, color=g_color).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[lo, hi]), title="UTCI (°C)"),
            y=alt.Y("y:N", title=None))
        dots = alt.Chart(pts).mark_point(size=180, filled=True).encode(
            x="UTCI:Q",
            y=alt.Y("Μοντέλο:N", title=None, sort=["GPR", "Τύπος"]),
            color=alt.Color("χρώμα:N", scale=None),
            tooltip=["Μοντέλο", "UTCI"])
        st.altair_chart((rule + dots).properties(height=120), width="stretch")
    except Exception:
        st.write(f"GPR: {gpr:.1f} ± {std:.1f} °C  |  Τύπος: {official:.1f} °C")

    # Honesty note — το σωστό framing.
    r2, rmse = bundle["metrics"]["r2"], bundle["metrics"]["rmse"]
    st.caption(
        f"Μοντέλο: R²={r2:.2f}, RMSE={rmse:.2f}°C στο test set "
        f"(vs naive persistence RMSE≈5.77°C — το GPR κερδίζει ~26%). "
        f"Το RMSE ~4°C σημαίνει ότι συχνά η πρόβλεψη πέφτει στην κατηγορία-γείτονα· "
        f"το official UTCI, χτισμένο πάνω στο NWP forecast του Open-Meteo, είναι "
        f"γενικά ακριβέστερο. Το Version B δείχνει την *αναπαραγωγή της μεθόδου* "
        f"με ποσοτικοποιημένη αβεβαιότητα.")

    with st.expander("🔧 Δες τα 30 ανακατασκευασμένα inputs"):
        st.dataframe(pd.DataFrame([out["features"]]).T.rename(columns={0: "τιμή"}))


# ── Standalone test mode ─────────────────────────────────────────────────────
# ΣΗΜΕΙΩΣΗ: το city search εδώ είναι ΜΟΝΟ για το μεμονωμένο τεστ, ώστε να έχει
# το feel του Version A. Στο πραγματικό integration χρησιμοποίησε το ΥΠΑΡΧΟΝ
# city search του Version A και κάλεσε render_gpr_forecast(lat, lon, sun).
if __name__ == "__main__":
    st.set_page_config(page_title="UTCI — Πρόγνωση Θερμικής Δυσφορίας", layout="centered")
    st.title("🌡️ UTCI — Πρόγνωση Θερμικής Δυσφορίας")

    city = st.text_input("🔎 Πόλη", value="Athens",
                         placeholder="π.χ. Athens, Salamina, Thessaloniki…")
    try:
        results = geocode_city(city)
    except Exception as e:
        st.error(f"Geocoding απέτυχε: {e}")
        results = []

    if results:
        def _label(i: int) -> str:
            r = results[i]
            bits = [r.get("name", "")]
            if r.get("admin1"):
                bits.append(r["admin1"])
            if r.get("country"):
                bits.append(r["country"])
            return ", ".join(b for b in bits if b)

        idx = st.selectbox("Τοποθεσία", range(len(results)), format_func=_label)
        chosen = results[idx]
        lat, lon = chosen["latitude"], chosen["longitude"]
        sun = st.toggle("Ήλιος (sun MRT)", value=True)
        st.caption(f"📍 {lat:.4f}, {lon:.4f}")
        render_gpr_forecast(lat, lon, sun)
    else:
        st.info("Γράψε το όνομα μιας πόλης για να ξεκινήσεις.")
