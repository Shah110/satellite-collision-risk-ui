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
# Sidebar: Links + Settings
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

st.sidebar.divider()
st.sidebar.subheader("🧾 Feature Name Display")
FEATURE_MAP_FILE = st.sidebar.text_input("Feature map file (optional)", value="feature_map.csv")
st.sidebar.caption("Create feature_map.csv with columns: model_feature, original_feature")

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

def display_name(col: str, fmap: dict) -> str:
    return fmap.get(str(col), str(col))

def display_names(cols, fmap: dict):
    return [display_name(c, fmap) for c in cols]

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
        return "#ff4d4f"  # red
    if 24 <= hours_to_tca <= 72:
        return "#faad14"  # yellow
    return "#52c41a"      # green

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

# ==========================================================
# Countdown (C)
# ==========================================================
st.header("⏱️ TCA Countdown")
make_countdown_card(tca_dt)

# ==========================================================
# Load Dataset
# ==========================================================
st.header("📊 Dataset")

df = None
if use_uploader:
    uploaded = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue, or disable uploader in the sidebar to use the repo CSV.")
    else:
        df = pd.read_csv(uploaded)
else:
    if not Path(DATA_FILE).exists():
        st.error(f"Dataset file not found: `{DATA_FILE}`. Upload it to the repo root or correct the name in sidebar.")
        st.stop()
    df = load_csv(DATA_FILE)

if df is None:
    st.stop()

st.write("**Shape:**", df.shape)
with st.expander("Show dataset preview", expanded=True):
    st.dataframe(df.head(30), use_container_width=True)

# ==========================================================
# Load Model + Feature Mapping
# ==========================================================
st.header("🧩 Model")

if not Path(MODEL_FILE).exists():
    st.error(f"Model file not found: `{MODEL_FILE}`. Upload it to the repo root or correct the name in sidebar.")
    st.stop()

try:
    model = load_model(MODEL_FILE)
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

feature_map = {}
if FEATURE_MAP_FILE and Path(FEATURE_MAP_FILE).exists():
    try:
        feature_map = load_feature_map(FEATURE_MAP_FILE)
        st.info(f"✅ Feature mapping loaded from `{FEATURE_MAP_FILE}`")
    except Exception as e:
        st.warning(f"⚠️ Found `{FEATURE_MAP_FILE}` but failed to load: {e}")
else:
    st.warning("ℹ️ No feature_map.csv found. UI will show model feature names (feature_0...).")

if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    with st.expander("Model expected features (Original Names)", expanded=False):
        st.dataframe(
            pd.DataFrame({
                "Model Feature": expected,
                "Original Name": display_names(expected, feature_map)
            }),
            use_container_width=True
        )

# ==========================================================
# Prediction Panel
# ==========================================================
st.header("🚀 Prediction Panel")

X = df.drop(columns=[TARGET_COL], errors="ignore")
X = safe_numeric(X)
X_aligned = align_features_for_model(model, X)

missing_cols, extra_cols = [], []
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    provided = list(X.columns)
    missing_cols = [c for c in expected if c not in provided]
    extra_cols = [c for c in provided if c not in expected]

m1, m2, m3 = st.columns(3)
m1.metric("Provided numeric features", len(X.columns))
m2.metric("Missing vs model", len(missing_cols))
m3.metric("Extra vs model", len(extra_cols))

if missing_cols:
    st.warning(f"Missing columns (filled with 0) (up to 25): {display_names(missing_cols, feature_map)[:25]}")
if extra_cols:
    st.info(f"Extra columns ignored (up to 25): {display_names(extra_cols, feature_map)[:25]}")

predict = st.button("🧠 Predict Collision Risk", type="primary")
if predict:
    try:
        preds = model.predict(X_aligned)
        out = df.copy()
        out["predicted_risk"] = preds
        st.success("Predictions generated ✅")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="collision_risk_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Most common cause: feature mismatch or missing preprocessing pipeline.")

# ==========================================================
# A + B: Maneuver Simulator + Uncertainty Ellipsoids (3D)
# ==========================================================
st.header("🧭 Maneuver Simulator + Uncertainty (3D)")

st.caption(
    "This is a **simple relative-motion demo** around TCA: r(t)=r0+v0*t. "
    "If your CSV has real state/covariance columns, we can wire them in."
)

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
    st.write("Pick relative state columns if they exist in your CSV (we will show original names too).")

    def pick_col(label, default_name):
        candidates = [c for c in cols if c.lower() == default_name.lower()]
        default = candidates[0] if candidates else None
        options = [None] + cols
        idx = 0 if default is None else (1 + cols.index(default))
        return st.selectbox(label, options=options, index=idx)

    c1, c2 = st.columns(2)
    with c1:
        rx_col = pick_col("Relative X (km) column", state_cols_default["rx"])
        ry_col = pick_col("Relative Y (km) column", state_cols_default["ry"])
        rz_col = pick_col("Relative Z (km) column", state_cols_default["rz"])
    with c2:
        vx_col = pick_col("Relative Vx (km/s) column", state_cols_default["vx"])
        vy_col = pick_col("Relative Vy (km/s) column", state_cols_default["vy"])
        vz_col = pick_col("Relative Vz (km/s) column", state_cols_default["vz"])

row_idx = st.number_input("Select conjunction row index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)

r0_demo = np.array([0.3, 0.2, 0.05])
v0_demo = np.array([-0.0006, 0.0004, 0.0])

row = df.iloc[int(row_idx)]

def get_state():
    try:
        if all([rx_col, ry_col, rz_col, vx_col, vy_col, vz_col]):
            r0 = np.array([float(row[rx_col]), float(row[ry_col]), float(row[rz_col])])
            v0 = np.array([float(row[vx_col]), float(row[vy_col]), float(row[vz_col])])
            return r0, v0, True
    except Exception:
        pass
    return r0_demo, v0_demo, False

r0_km, v0_kms, using_csv = get_state()
st.info("Using CSV state ✅" if using_csv else "Using DEMO state (no relative state columns selected).")

span_hours = st.slider("Time window around TCA (hours)", 0.5, 12.0, 2.0, 0.5)
N = 160
t = np.linspace(-span_hours*3600, span_hours*3600, N)

r_nom = propagate_relative_motion(r0_km, v0_kms, t)

dv_kms = delta_v / 1000.0
v_man = apply_delta_v(v0_kms, dv_kms, burn_direction)
r_man = propagate_relative_motion(r0_km, v_man, t)

miss_nom_km = np.linalg.norm(r_nom, axis=1)
miss_man_km = np.linalg.norm(r_man, axis=1)

tca_idx = np.argmin(np.abs(t))
md_nom = miss_nom_km[tca_idx]
md_man = miss_man_km[tca_idx]

k1, k2, k3 = st.columns(3)
k1.metric("Miss Distance @TCA (Nominal)", f"{md_nom*1000:.1f} m")
k2.metric("Miss Distance @TCA (Maneuvered)", f"{md_man*1000:.1f} m")
k3.metric("Increase", f"{(md_man-md_nom)*1000:.1f} m")

cov1 = default_cov(km_sigma=0.2)
cov2 = default_cov(km_sigma=0.2)

satA_center = np.array([0.0, 0.0, 0.0])
satB_center_nom = r_nom[tca_idx]
satB_center_man = r_man[tca_idx]

x1, y1, z1 = ellipsoid_mesh(satA_center, cov1, n=26, k=3.0)
x2n, y2n, z2n = ellipsoid_mesh(satB_center_nom, cov2, n=26, k=3.0)
x2m, y2m, z2m = ellipsoid_mesh(satB_center_man, cov2, n=26, k=3.0)

sigA = np.sqrt(np.max(np.linalg.eigvalsh(cov1))) * 3.0
sigB = np.sqrt(np.max(np.linalg.eigvalsh(cov2))) * 3.0
overlap_nom = (np.linalg.norm(satB_center_nom - satA_center) < (sigA + sigB))
overlap_man = (np.linalg.norm(satB_center_man - satA_center) < (sigA + sigB))

st.write(
    f"**Ellipsoid overlap risk (heuristic):** "
    f"Nominal = {'⚠️ High' if overlap_nom else '✅ Low'} | "
    f"Maneuvered = {'⚠️ High' if overlap_man else '✅ Low'}"
)

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=r_nom[:, 0], y=r_nom[:, 1], z=r_nom[:, 2],
    mode="lines",
    name="Nominal Path"
))

fig.add_trace(go.Scatter3d(
    x=r_man[:, 0], y=r_man[:, 1], z=r_man[:, 2],
    mode="lines",
    name="Maneuvered Path (ΔV)"
))

fig.add_trace(go.Scatter3d(
    x=[satB_center_nom[0]], y=[satB_center_nom[1]], z=[satB_center_nom[2]],
    mode="markers",
    name="TCA point (Nominal)",
    marker=dict(size=5)
))
fig.add_trace(go.Scatter3d(
    x=[satB_center_man[0]], y=[satB_center_man[1]], z=[satB_center_man[2]],
    mode="markers",
    name="TCA point (Maneuvered)",
    marker=dict(size=5)
))

fig.add_trace(go.Surface(
    x=x1, y=y1, z=z1,
    name="Sat A Uncertainty (3σ)",
    showscale=False,
    opacity=0.18
))

fig.add_trace(go.Surface(
    x=x2n, y=y2n, z=z2n,
    name="Sat B Uncertainty @TCA (Nominal, 3σ)",
    showscale=False,
    opacity=0.18
))

fig.add_trace(go.Surface(
    x=x2m, y=y2m, z=z2m,
    name="Sat B Uncertainty @TCA (Maneuvered, 3σ)",
    showscale=False,
    opacity=0.18
))

fig.update_layout(
    height=650,
    scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)",
        aspectmode="data"
    ),
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

if show_debug:
    st.subheader("Debug")
    st.write("r0_km:", r0_km)
    st.write("v0_kms:", v0_kms)
    st.write("v_man_kms:", v_man)
    st.write("ΔV (m/s):", delta_v)
