import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go

# ==========================================================
# Page Config
# ==========================================================
st.set_page_config(page_title="Satellite Collision Risk UI", layout="wide")
st.title("🛰️ Satellite Collision Risk UI")
st.success("App is running ✅")

# ==========================================================
# Sidebar: Links + Global Settings
# ==========================================================
st.sidebar.header("🚀 Project Notebooks (Colab)")
st.sidebar.markdown(
    "- [Phase 1](https://colab.research.google.com/drive/1Utaq_FtgsHhV215frPX-CQhC6nMwkfs3?usp=sharing)\n"
    "- [Phase 2](https://colab.research.google.com/drive/1oog4BFnxr5ohss8HHJszAzv2rZqWFJO3?usp=sharing)\n"
    "- [Phase 3](https://colab.research.google.com/drive/1DGncV8CkSyTKbe5YBfqrTC-qrQxwfau-?usp=sharing)"
)
st.sidebar.divider()
st.sidebar.subheader("⚙️ Data / Model Settings")

DATA_FILE = st.sidebar.text_input("Dataset file in repo", value="combined_data.csv")
MODEL_FILE = st.sidebar.text_input("Model file in repo", value="collision_risk_model.pkl")
TARGET_COL = st.sidebar.text_input("Target column (if exists)", value="risk")

use_uploader = st.sidebar.toggle("Use CSV uploader instead of repo file", value=True)
show_debug = st.sidebar.toggle("Show debug info", value=False)

st.sidebar.divider()
st.sidebar.subheader("🧾 Feature Name Display")
FEATURE_MAP_FILE = st.sidebar.text_input("Feature map file (optional)", value="feature_map.csv")
st.sidebar.caption("feature_map.csv must have columns: model_feature, original_feature")

st.sidebar.divider()
st.sidebar.subheader("🚨 Alert Threshold")
risk_threshold = st.sidebar.slider("Risk probability alert threshold", 0.0, 1.0, 0.50, 0.01)

st.sidebar.divider()
st.sidebar.subheader("🛰️ Conjunction Ops Controls")

# --- Delta-V slider
delta_v = st.sidebar.slider("ΔV burn magnitude (m/s)", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
burn_direction = st.sidebar.selectbox(
    "ΔV direction (in relative frame)",
    ["Along-track (V direction)", "Radial (X)", "Cross-track (Z)"]
)

# --- TCA datetime
now_local = datetime.now()
tca_date = st.sidebar.date_input("TCA date", value=now_local.date())
tca_time = st.sidebar.time_input("TCA time", value=(now_local + timedelta(hours=48)).time())
tca_dt = datetime.combine(tca_date, tca_time)

# ==========================================================
# Helpers
# ==========================================================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_feature_map(path: str) -> dict:
    fm = pd.read_csv(path)
    if not {"model_feature", "original_feature"}.issubset(fm.columns):
        raise ValueError("feature_map.csv must contain columns: model_feature, original_feature")
    return dict(zip(fm["model_feature"].astype(str), fm["original_feature"].astype(str)))

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")
    return X.select_dtypes(include=["number"]).fillna(0)

def align_features_for_model(model, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        return X.reindex(columns=expected, fill_value=0)
    return X

# ✅ One-shot DISPLAY rename (UI only)
def rename_for_display(df_in: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    if not fmap:
        return df_in
    rename_dict = {k: v for k, v in fmap.items() if k in df_in.columns}
    return df_in.rename(columns=rename_dict)

def get_probabilities(model, X_aligned: pd.DataFrame):
    """
    Returns a 1D probability array for 'positive/risk' class if possible, else None.
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_aligned)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            if isinstance(proba, np.ndarray) and proba.ndim == 1:
                return proba
        except Exception:
            return None
    return None

def format_countdown(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    if total < 0:
        total = abs(total)
        sign = "-"
    else:
        sign = ""
    days = total // 86400
    hrs = (total % 86400) // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{sign}{days:02d}d {hrs:02d}h {mins:02d}m {secs:02d}s"

def countdown_color(hours_to_tca: float) -> str:
    if hours_to_tca < 24:
        return "#ff4d4f"
    if 24 <= hours_to_tca <= 72:
        return "#faad14"
    return "#52c41a"

def make_countdown_card(tca_dt: datetime):
    now = datetime.now()
    delta = tca_dt - now
    hours_to_tca = delta.total_seconds() / 3600.0

    color = countdown_color(hours_to_tca)
    label = "RED: < 24h (DECIDE NOW)" if hours_to_tca < 24 else ("YELLOW: 24–72h (MONITOR)" if hours_to_tca <= 72 else "GREEN: > 72h (PLAN)")
    val = format_countdown(delta)

    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.15);
            background: rgba(0,0,0,0.03);
            ">
            <div style="font-size: 14px; opacity: 0.8;">⏱️ Time to Closest Approach (TCA)</div>
            <div style="font-size: 34px; font-weight: 800; color: {color}; margin-top: 6px;">{val}</div>
            <div style="font-size: 13px; margin-top: 4px;">{label}</div>
            <div style="font-size: 12px; opacity: 0.7; margin-top: 6px;">TCA: {tca_dt.strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def ellipsoid_mesh(center, cov, n=24, k=3.0):
    cov = np.array(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    radii = k * np.sqrt(vals)

    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    xyz = np.stack([x, y, z], axis=-1)
    scaled = xyz @ np.diag(radii)
    rotated = scaled @ vecs.T
    pts = rotated + np.array(center)

    return pts[..., 0], pts[..., 1], pts[..., 2]

def propagate_relative_motion(r0, v0, t_seconds):
    t = np.asarray(t_seconds).reshape(-1, 1)
    return r0.reshape(1, 3) + v0.reshape(1, 3) * t

def apply_delta_v(v0, dv, direction):
    v0 = np.array(v0, dtype=float)
    if direction.startswith("Along-track"):
        norm = np.linalg.norm(v0)
        unit = v0 / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    elif direction.startswith("Radial"):
        unit = np.array([1.0, 0.0, 0.0])
    else:
        unit = np.array([0.0, 0.0, 1.0])
    return v0 + dv * unit

def default_cov(km_sigma=0.2):
    s2 = km_sigma ** 2
    return np.diag([s2, s2, s2])

def build_event_labels(df: pd.DataFrame) -> pd.Series:
    """
    Creates a human-friendly event label for each row (no extra user inputs).
    Tries common columns, otherwise falls back to index.
    """
    cols = set([c.lower() for c in df.columns])

    # Try best columns if present
    if "event_id" in cols:
        c = [x for x in df.columns if x.lower() == "event_id"][0]
        return df[c].astype(str)

    # Satellite IDs if present
    sat_a_candidates = ["sat_a", "satellite_a", "primary_sat", "sat1", "object1", "norad_a"]
    sat_b_candidates = ["sat_b", "satellite_b", "secondary_sat", "sat2", "object2", "norad_b"]
    tca_candidates = ["tca", "tca_time", "tca_datetime", "time_of_closest_approach"]

    sat_a = next((c for c in df.columns if c.lower() in sat_a_candidates), None)
    sat_b = next((c for c in df.columns if c.lower() in sat_b_candidates), None)
    tca_c = next((c for c in df.columns if c.lower() in tca_candidates), None)

    if sat_a and sat_b and tca_c:
        return (df[sat_a].astype(str) + " vs " + df[sat_b].astype(str) + " @ " + df[tca_c].astype(str))

    if sat_a and sat_b:
        return (df[sat_a].astype(str) + " vs " + df[sat_b].astype(str))

    # fallback
    return pd.Series([f"Event #{i}" for i in range(len(df))], index=df.index)

def action_recommendation(prob: float | None, hours_to_tca: float) -> str:
    # If no probability, recommend based on time only
    if prob is None:
        if hours_to_tca < 24:
            return "🔴 **DECIDE NOW:** Request urgent update + prepare maneuver."
        if hours_to_tca <= 72:
            return "🟡 **MONITOR:** Track updates, pre-plan maneuver options."
        return "🟢 **PLAN:** Normal monitoring, plan if trend worsens."

    # With probability + time
    if hours_to_tca < 24:
        return "🔴 **DECIDE NOW:** Execute/approve maneuver if risk remains high."
    if hours_to_tca <= 72:
        return "🟡 **MONITOR:** Refine OD, coordinate, prepare maneuver plan."
    return "🟢 **PLAN:** Routine monitoring, schedule updates."

def kpi_badge(prob: float | None, threshold: float):
    if prob is None:
        st.info("Model probability not available (no predict_proba). Showing class/score only.")
        return
    if prob >= threshold:
        st.error(f"⚠️ ALERT: Risk probability {prob:.3f} ≥ threshold {threshold:.2f}")
    else:
        st.success(f"✅ OK: Risk probability {prob:.3f} < threshold {threshold:.2f}")

# ==========================================================
# Session state for scenario saving
# ==========================================================
if "scenarios" not in st.session_state:
    st.session_state.scenarios = []  # list of dicts

# ==========================================================
# Load Dataset
# ==========================================================
df = None
if use_uploader:
    uploaded = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    if Path(DATA_FILE).exists():
        df = load_csv(DATA_FILE)

if df is None:
    st.warning("Upload a CSV (or disable uploader and ensure combined_data.csv exists in repo).")
    st.stop()

# ==========================================================
# Load Model + Feature Map
# ==========================================================
if not Path(MODEL_FILE).exists():
    st.error(f"Model file not found: `{MODEL_FILE}`. Upload it to the repo root or correct the name in sidebar.")
    st.stop()

try:
    model = load_model(MODEL_FILE)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Fix: add required libraries to requirements.txt (e.g., scikit-learn).")
    st.stop()

feature_map = {}
if FEATURE_MAP_FILE and Path(FEATURE_MAP_FILE).exists():
    try:
        feature_map = load_feature_map(FEATURE_MAP_FILE)
    except Exception as e:
        st.warning(f"⚠️ Found `{FEATURE_MAP_FILE}` but failed to load: {e}")

# ==========================================================
# Build event labels + selector
# ==========================================================
event_labels = build_event_labels(df)
label_to_index = {lbl: idx for lbl, idx in zip(event_labels.tolist(), df.index.tolist())}

st.sidebar.divider()
st.sidebar.subheader("📌 Select Conjunction Event")
selected_label = st.sidebar.selectbox("Choose event", options=list(label_to_index.keys()))
row_idx = label_to_index[selected_label]
row = df.loc[row_idx]

# ==========================================================
# Countdown header (C)
# ==========================================================
st.header("⏱️ TCA Countdown")
make_countdown_card(tca_dt)

# ==========================================================
# Tabs (better UX)
# ==========================================================
tab_dataset, tab_prediction, tab_maneuver, tab_reports = st.tabs(
    ["📊 Dataset", "🚀 Prediction", "🧭 Maneuver + Uncertainty", "🧾 Reports"]
)

# ==========================================================
# TAB 1: Dataset
# ==========================================================
with tab_dataset:
    st.subheader("Dataset Overview")
    st.write("**Shape:**", df.shape)

    display_df = rename_for_display(df, feature_map)

    left, right = st.columns([2, 1])
    with left:
        with st.expander("Preview (Original Names if mapping exists)", expanded=True):
            st.dataframe(display_df.head(50), use_container_width=True)
    with right:
        st.write("**Selected Event:**", selected_label)
        st.write("**Row index:**", row_idx)

    # Distribution of probabilities if available
    X_all = df.drop(columns=[TARGET_COL], errors="ignore")
    X_all = safe_numeric(X_all)
    X_all_aligned = align_features_for_model(model, X_all)
    probs_all = get_probabilities(model, X_all_aligned)

    if probs_all is not None:
        st.subheader("Risk Probability Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=probs_all, nbinsx=30, name="Risk Probability"))
        fig_hist.update_layout(
            height=320,
            xaxis_title="Risk Probability",
            yaxis_title="Count",
            bargap=0.05
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Probability distribution not available (model has no predict_proba).")

# ==========================================================
# TAB 2: Prediction
# ==========================================================
with tab_prediction:
    st.subheader("Prediction for Selected Event")

    # Build single-row X for selected event
    single_df = pd.DataFrame([row])
    X1 = single_df.drop(columns=[TARGET_COL], errors="ignore")
    X1 = safe_numeric(X1)
    X1_aligned = align_features_for_model(model, X1)

    # Predict class/value
    pred = None
    prob = None
    try:
        pred = model.predict(X1_aligned)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    probs = get_probabilities(model, X1_aligned)
    if probs is not None:
        prob = float(probs[0])

    # KPIs
    hours_to_tca = (tca_dt - datetime.now()).total_seconds() / 3600.0
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Predicted Risk (class/value)", f"{pred}")
    with c2:
        st.metric("Risk Probability", "N/A" if prob is None else f"{prob:.3f}")
    with c3:
        st.metric("Hours to TCA", f"{hours_to_tca:.1f}")
    with c4:
        st.metric("Alert Threshold", f"{risk_threshold:.2f}")

    kpi_badge(prob, risk_threshold)

    st.markdown("### Operator Recommendation")
    st.write(action_recommendation(prob, hours_to_tca))

    st.markdown("### Selected Event Data (display)")
    st.dataframe(rename_for_display(single_df, feature_map), use_container_width=True)

    # Model expected features table
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        exp_table = pd.DataFrame(
            {"Model Feature": expected, "Original Name": [feature_map.get(c, c) for c in expected]}
        )
        with st.expander("Model expected features (Original Names)", expanded=False):
            st.dataframe(exp_table, use_container_width=True)

# ==========================================================
# TAB 3: Maneuver + Uncertainty (A + B)
# ==========================================================
with tab_maneuver:
    st.subheader("Maneuver Simulator + Error Ellipsoids (3D)")
    st.caption(
        "This uses a **simple linear relative-motion demo** around TCA: r(t)=r0+v0*t. "
        "Your ML model prediction is computed from the dataset features; the maneuver plot is a separate visualization."
    )

    # Allow mapping to real columns if present
    cols = list(df.columns)
    state_cols_default = {
        "rx": "rel_x_km",
        "ry": "rel_y_km",
        "rz": "rel_z_km",
        "vx": "rel_vx_kms",
        "vy": "rel_vy_kms",
        "vz": "rel_vz_kms",
    }

    with st.expander("State configuration (optional)", expanded=False):
        st.write("Select relative state columns if they exist in your CSV; otherwise demo values are used.")

        def pick_col(label, default_name):
            candidates = [c for c in cols if c.lower() == default_name.lower()]
            default = candidates[0] if candidates else None
            options = [None] + cols
            idx = 0 if default is None else (1 + cols.index(default))
            return st.selectbox(label, options=options, index=idx)

        cc1, cc2 = st.columns(2)
        with cc1:
            rx_col = pick_col("Relative X (km) column", state_cols_default["rx"])
            ry_col = pick_col("Relative Y (km) column", state_cols_default["ry"])
            rz_col = pick_col("Relative Z (km) column", state_cols_default["rz"])
        with cc2:
            vx_col = pick_col("Relative Vx (km/s) column", state_cols_default["vx"])
            vy_col = pick_col("Relative Vy (km/s) column", state_cols_default["vy"])
            vz_col = pick_col("Relative Vz (km/s) column", state_cols_default["vz"])

    # Demo fallback state (km, km/s)
    r0_demo = np.array([0.3, 0.2, 0.05])        # 300m, 200m, 50m
    v0_demo = np.array([-0.0006, 0.0004, 0.0])  # km/s (0.6 m/s, 0.4 m/s)

    def get_state_from_row(r):
        try:
            if all([rx_col, ry_col, rz_col, vx_col, vy_col, vz_col]):
                r0 = np.array([float(r[rx_col]), float(r[ry_col]), float(r[rz_col])])
                v0 = np.array([float(r[vx_col]), float(r[vy_col]), float(r[vz_col])])
                return r0, v0, True
        except Exception:
            pass
        return r0_demo, v0_demo, False

    r0_km, v0_kms, using_csv = get_state_from_row(row)
    st.info("Using CSV state ✅" if using_csv else "Using DEMO state (no relative state columns selected).")

    span_hours = st.slider("Time window around TCA (hours)", 0.5, 12.0, 2.0, 0.5)
    N = 160
    t = np.linspace(-span_hours * 3600, span_hours * 3600, N)  # seconds

    r_nom = propagate_relative_motion(r0_km, v0_kms, t)

    dv_kms = delta_v / 1000.0  # m/s -> km/s
    v_man = apply_delta_v(v0_kms, dv_kms, burn_direction)
    r_man = propagate_relative_motion(r0_km, v_man, t)

    miss_nom_km = np.linalg.norm(r_nom, axis=1)
    miss_man_km = np.linalg.norm(r_man, axis=1)

    tca_idx = int(np.argmin(np.abs(t)))
    md_nom = float(miss_nom_km[tca_idx])
    md_man = float(miss_man_km[tca_idx])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Miss Distance @TCA (Nominal)", f"{md_nom * 1000:.1f} m")
    k2.metric("Miss Distance @TCA (Maneuvered)", f"{md_man * 1000:.1f} m")
    k3.metric("Increase", f"{(md_man - md_nom) * 1000:.1f} m")
    k4.metric("ΔV", f"{delta_v:.2f} m/s")

    # Uncertainty ellipsoids
    cov1 = default_cov(km_sigma=0.2)
    cov2 = default_cov(km_sigma=0.2)

    satA_center = np.array([0.0, 0.0, 0.0])
    satB_center_nom = r_nom[tca_idx]
    satB_center_man = r_man[tca_idx]

    x1, y1, z1 = ellipsoid_mesh(satA_center, cov1, n=26, k=3.0)
    x2n, y2n, z2n = ellipsoid_mesh(satB_center_nom, cov2, n=26, k=3.0)
    x2m, y2m, z2m = ellipsoid_mesh(satB_center_man, cov2, n=26, k=3.0)

    sigA = float(np.sqrt(np.max(np.linalg.eigvalsh(cov1))) * 3.0)
    sigB = float(np.sqrt(np.max(np.linalg.eigvalsh(cov2))) * 3.0)
    overlap_nom = (np.linalg.norm(satB_center_nom - satA_center) < (sigA + sigB))
    overlap_man = (np.linalg.norm(satB_center_man - satA_center) < (sigA + sigB))

    st.write(
        f"**Ellipsoid overlap risk (heuristic):** "
        f"Nominal = {'⚠️ High' if overlap_nom else '✅ Low'} | "
        f"Maneuvered = {'⚠️ High' if overlap_man else '✅ Low'}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=r_nom[:, 0], y=r_nom[:, 1], z=r_nom[:, 2], mode="lines", name="Nominal Path"))
    fig.add_trace(go.Scatter3d(x=r_man[:, 0], y=r_man[:, 1], z=r_man[:, 2], mode="lines", name="Maneuvered Path (ΔV)"))

    fig.add_trace(go.Scatter3d(
        x=[satB_center_nom[0]], y=[satB_center_nom[1]], z=[satB_center_nom[2]],
        mode="markers", name="TCA (Nominal)", marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter3d(
        x=[satB_center_man[0]], y=[satB_center_man[1]], z=[satB_center_man[2]],
        mode="markers", name="TCA (Maneuvered)", marker=dict(size=5)
    ))

    fig.add_trace(go.Surface(x=x1, y=y1, z=z1, name="Sat A Uncertainty (3σ)", showscale=False, opacity=0.18))
    fig.add_trace(go.Surface(x=x2n, y=y2n, z=z2n, name="Sat B @TCA (Nominal, 3σ)", showscale=False, opacity=0.18))
    fig.add_trace(go.Surface(x=x2m, y=y2m, z=z2m, name="Sat B @TCA (Maneuvered, 3σ)", showscale=False, opacity=0.18))

    fig.update_layout(
        height=650,
        scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)", aspectmode="data"),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Save scenario
    st.divider()
    st.subheader("Save Scenario")
    st.caption("This saves maneuver parameters + miss distance results for comparison (visual simulator).")

    scenario_name = st.text_input("Scenario name", value=f"{selected_label} | ΔV {delta_v:.2f} m/s")
    if st.button("💾 Save current scenario"):
        st.session_state.scenarios.append({
            "event": selected_label,
            "scenario": scenario_name,
            "delta_v_mps": float(delta_v),
            "direction": burn_direction,
            "miss_nom_m": md_nom * 1000.0,
            "miss_man_m": md_man * 1000.0,
            "increase_m": (md_man - md_nom) * 1000.0,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        st.success("Scenario saved ✅")

    if show_debug:
        st.write("DEBUG r0_km:", r0_km)
        st.write("DEBUG v0_kms:", v0_kms)
        st.write("DEBUG v_man_kms:", v_man)

# ==========================================================
# TAB 4: Reports
# ==========================================================
with tab_reports:
    st.subheader("Scenario Comparison Report")

    if len(st.session_state.scenarios) == 0:
        st.info("No scenarios saved yet. Go to the Maneuver tab and click “Save current scenario”.")
    else:
        report_df = pd.DataFrame(st.session_state.scenarios)
        st.dataframe(report_df, use_container_width=True)

        # Quick best scenario per event (max increase)
        st.markdown("### Best Scenario (by miss distance increase) per event")
        best = report_df.sort_values("increase_m", ascending=False).groupby("event", as_index=False).head(1)
        st.dataframe(best, use_container_width=True)

        st.download_button(
            "⬇️ Download Scenario Report CSV",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name="scenario_report.csv",
            mime="text/csv",
        )

        if st.button("🧹 Clear all saved scenarios"):
            st.session_state.scenarios = []
            st.success("Cleared ✅")
