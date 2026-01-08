
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
DATA_FILE = st.sidebar.text_input("Dataset file in repo", value="cleaned_train_final.csv")
MODEL_FILE = st.sidebar.text_input("Model file in repo", value="collision_risk_model.pkl")
TARGET_COL = st.sidebar.text_input("Target column (if exists)", value="risk")

use_uploader = st.sidebar.toggle("Use CSV uploader instead of repo file", value=False)
show_debug = st.sidebar.toggle("Show debug panels", value=False)

st.sidebar.divider()
st.sidebar.subheader("🧾 Feature Mapping (IMPORTANT)")
FEATURE_MAP_FILE = st.sidebar.text_input("feature_map.csv", value="feature_map.csv")
use_feature_map = st.sidebar.toggle("Use feature_map.csv to build model input", value=True)
st.sidebar.caption("feature_map.csv columns: model_feature, original_feature")

st.sidebar.divider()
st.sidebar.subheader("🚨 Alert Threshold (Risk Score 0–1)")
risk_threshold = st.sidebar.slider("Risk score threshold", 0.0, 1.0, 0.50, 0.01)

st.sidebar.divider()
st.sidebar.subheader("🛰️ Conjunction Ops Controls")
delta_v = st.sidebar.slider("ΔV burn magnitude (m/s)", 0.0, 2.0, 0.0, 0.01)
burn_direction = st.sidebar.selectbox(
    "ΔV direction (relative frame)",
    ["Along-track (V direction)", "Radial (X)", "Cross-track (Z)"],
)

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

def safe_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.select_dtypes(include=["number"]).fillna(0)

def normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mn >= -1e-9 and mx <= 1.0 + 1e-9:
        return np.clip(x, 0.0, 1.0)
    if np.isclose(mx - mn, 0.0):
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def get_risk_output(model, X_aligned: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X_aligned))
            if p.ndim == 2 and p.shape[1] >= 2:
                return "proba", p[:, 1]
            if p.ndim == 1:
                return "proba", p
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X_aligned)
            return "score", np.asarray(s).reshape(-1)
        except Exception:
            pass

    y = model.predict(X_aligned)
    return "pred", np.asarray(y).reshape(-1)

def format_countdown(delta: timedelta) -> str:
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    days = total // 86400
    hrs = (total % 86400) // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    return f"{sign}{days:02d}d {hrs:02d}h {mins:02d}m {secs:02d}s"

def countdown_color(hours_to_tca: float) -> str:
    if hours_to_tca < 24:
        return "#ff4d4f"
    if hours_to_tca <= 72:
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
        <div style="padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,0.15);background:rgba(0,0,0,0.03);">
          <div style="font-size:14px;opacity:0.8;">⏱️ Time to Closest Approach (TCA)</div>
          <div style="font-size:34px;font-weight:800;color:{color};margin-top:6px;">{val}</div>
          <div style="font-size:13px;margin-top:4px;">{label}</div>
          <div style="font-size:12px;opacity:0.7;margin-top:6px;">TCA: {tca_dt.strftime("%Y-%m-%d %H:%M:%S")}</div>
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
    cols_lower = {c.lower(): c for c in df.columns}
    if "event_id" in cols_lower:
        return df[cols_lower["event_id"]].astype(str)
    return pd.Series([f"Event #{i}" for i in range(len(df))], index=df.index)

def operator_recommendation(hours_to_tca: float) -> str:
    if hours_to_tca < 24:
        return "🔴 **DECIDE NOW:** Request urgent tracking update + prepare/execute maneuver if needed."
    if hours_to_tca <= 72:
        return "🟡 **MONITOR:** Continue monitoring, coordinate, and pre-plan maneuver options."
    return "🟢 **PLAN:** Normal monitoring and planning."

def show_risk_badge(score_0_1: float):
    if score_0_1 >= risk_threshold:
        st.error(f"⚠️ ALERT: Risk Score {score_0_1:.3f} ≥ {risk_threshold:.2f}")
    else:
        st.success(f"✅ OK: Risk Score {score_0_1:.3f} < {risk_threshold:.2f}")

def try_infer_mapping_by_numeric_order(df: pd.DataFrame, expected: list[str]) -> dict:
    numeric_cols = list(safe_numeric_only(df).columns)
    n = min(len(expected), len(numeric_cols))
    return {expected[i]: numeric_cols[i] for i in range(n)}

def build_model_input(df: pd.DataFrame, model, fmap: dict | None, target_col: str):
    debug = {}
    X_src = df.drop(columns=[target_col], errors="ignore")

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        debug["expected_count"] = len(expected)
        debug["expected"] = expected

        if all(c in X_src.columns for c in expected):
            Xnum = safe_numeric_only(X_src)
            X_aligned = Xnum.reindex(columns=expected, fill_value=0)
            debug["mode"] = "direct_match"
            debug["missing"] = [c for c in expected if c not in Xnum.columns]
            debug["extra"] = [c for c in Xnum.columns if c not in expected]
            return X_aligned, debug

        if fmap:
            X_out = pd.DataFrame(index=df.index)
            missing_from_csv = []
            for model_col in expected:
                orig_col = fmap.get(model_col)
                if orig_col is None:
                    X_out[model_col] = 0.0
                    missing_from_csv.append(f"(no map) {model_col}")
                    continue
                if orig_col not in X_src.columns:
                    X_out[model_col] = 0.0
                    missing_from_csv.append(orig_col)
                else:
                    X_out[model_col] = pd.to_numeric(X_src[orig_col], errors="coerce").fillna(0.0)

            debug["mode"] = "feature_map"
            debug["missing"] = missing_from_csv
            debug["extra"] = [c for c in X_src.columns if c not in set(fmap.values())]
            return X_out, debug

        inferred = try_infer_mapping_by_numeric_order(X_src, expected)
        X_out = pd.DataFrame(index=df.index)
        for model_col in expected:
            orig_col = inferred.get(model_col)
            if orig_col is None:
                X_out[model_col] = 0.0
            else:
                X_out[model_col] = pd.to_numeric(X_src[orig_col], errors="coerce").fillna(0.0)

        debug["mode"] = "inferred_numeric_order"
        debug["inferred_map_preview"] = dict(list(inferred.items())[:10])
        return X_out, debug

    Xnum = safe_numeric_only(X_src)
    debug["mode"] = "no_feature_names_in_model"
    debug["provided_numeric_cols"] = list(Xnum.columns)
    return Xnum, debug

# ==========================================================
# Load Dataset
# ==========================================================
df = None
if use_uploader:
    up = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
else:
    if not Path(DATA_FILE).exists():
        st.error(f"Dataset file not found: `{DATA_FILE}`. Upload it to repo or correct in sidebar.")
        st.stop()
    df = load_csv(DATA_FILE)

if df is None:
    st.warning("Upload a CSV (or disable uploader and ensure the repo CSV exists).")
    st.stop()

# ==========================================================
# Load Model
# ==========================================================
if not Path(MODEL_FILE).exists():
    st.error(f"Model file not found: `{MODEL_FILE}`. Upload it to repo or correct in sidebar.")
    st.stop()

try:
    model = load_model(MODEL_FILE)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("If you see 'No module named sklearn', add scikit-learn to requirements.txt and reboot Streamlit.")
    st.stop()

# ==========================================================
# Load Feature Map
# ==========================================================
feature_map = None
if use_feature_map and FEATURE_MAP_FILE and Path(FEATURE_MAP_FILE).exists():
    try:
        feature_map = load_feature_map(FEATURE_MAP_FILE)
        st.sidebar.success("feature_map.csv loaded ✅")
    except Exception as e:
        st.sidebar.error(f"feature_map.csv error: {e}")
        feature_map = None
elif use_feature_map:
    st.sidebar.warning("feature_map.csv not found. If your model expects feature_0.., outputs may be constant.")

# ==========================================================
# Event selector
# ==========================================================
event_labels = build_event_labels(df)
label_to_index = {lbl: idx for lbl, idx in zip(event_labels.tolist(), df.index.tolist())}

st.sidebar.divider()
st.sidebar.subheader("📌 Select Conjunction Event")
selected_label = st.sidebar.selectbox("Choose event", options=list(label_to_index.keys()))
row_idx = label_to_index[selected_label]
row = df.loc[row_idx]

# ==========================================================
# Countdown
# ==========================================================
#st.header("⏱️ TCA Countdown")
#make_countdown_card(tca_dt)

# ==========================================================
# Tabs
# ==========================================================
tab_dataset, tab_prediction, tab_maneuver, tab_reports = st.tabs(
    ["📊 Dataset", "🚀 Prediction", "🧭 Maneuver + Uncertainty", "🧾 Reports"]
)

# ==========================================================
# TAB 1: Dataset
# ==========================================================
with tab_dataset:
    st.subheader("Dataset Preview")
    st.write("**File:**", DATA_FILE if not use_uploader else "Uploaded CSV")
    st.write("**Shape:**", df.shape)
    with st.expander("Show dataset preview", expanded=True):
        st.dataframe(df.head(50), use_container_width=True)

    X_all_aligned, dbg_all = build_model_input(df, model, feature_map, TARGET_COL)
    kind_all, values_all = get_risk_output(model, X_all_aligned)

    values_all = np.asarray(values_all, dtype=float).reshape(-1)
    risk_all = normalize_0_1(values_all)

    st.subheader("Risk Output Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model output type", kind_all)
    c2.metric("Rows", f"{len(values_all):,}")
    c3.metric("Unique raw outputs", f"{len(np.unique(values_all)):,}")
    c4.metric("Std (raw)", f"{np.std(values_all):.6f}")

    is_flat = (len(np.unique(values_all)) <= 2) or (np.isclose(np.std(values_all), 0.0))

    st.subheader("Risk Score Distribution (0–1)")
    if len(np.unique(risk_all)) == 1:
        st.info(f"All risk scores are the same: **{float(risk_all[0]):.6f}**")
    else:
        nbins = 50 if len(risk_all) > 50000 else 30
        fig = go.Figure(data=[go.Histogram(x=risk_all, nbinsx=nbins)])
        fig.update_layout(height=360, xaxis_title="Risk Score (0–1)", yaxis_title="Count", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Input build mode: **{dbg_all.get('mode','?')}**")

    if show_debug:
        with st.expander("Debug: Model input wiring", expanded=False):
            st.write(dbg_all)
            st.write("X_all_aligned shape:", X_all_aligned.shape)
            st.write("X_all_aligned head:", X_all_aligned.head())

# ==========================================================
# TAB 2: Prediction
# ==========================================================
with tab_prediction:
    st.subheader("Prediction for Selected Event")
    single_df = pd.DataFrame([row])

    X1_aligned, dbg1 = build_model_input(single_df, model, feature_map, TARGET_COL)
    kind1, values1 = get_risk_output(model, X1_aligned)

    raw_value = float(np.asarray(values1).reshape(-1)[0]) if len(values1) else 0.0
    risk_score = float(normalize_0_1(np.array([raw_value]))[0])

    hours_to_tca = (tca_dt - datetime.now()).total_seconds() / 3600.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score (0–1)", f"{risk_score:.3f}")
    m2.metric("Raw Model Output", f"{raw_value:.4f}")
    m3.metric("Output Type", kind1)
    m4.metric("Hours to TCA", f"{hours_to_tca:.1f}")

    show_risk_badge(risk_score)

    st.markdown("### Operator Recommendation")
    st.write(operator_recommendation(hours_to_tca))

    st.markdown("### Selected Event Row")
    st.dataframe(single_df, use_container_width=True)

    if show_debug:
        with st.expander("Debug: Selected Event wiring", expanded=False):
            st.write(dbg1)
            st.write("X1_aligned values:", X1_aligned.iloc[0].to_dict())

# ==========================================================
# TAB 3: Maneuver + Uncertainty
# ==========================================================
with tab_maneuver:
    st.subheader("Maneuver Simulator + Error Ellipsoids (3D)")
    st.caption(
        "This is a **visual demo** using linear relative motion around TCA: r(t)=r0+v0*t.\n"
        "It does not change ML predictions unless maneuver features are part of the model."
    )

    cols = list(df.columns)
    defaults = {
        "rx": "rel_x_km", "ry": "rel_y_km", "rz": "rel_z_km",
        "vx": "rel_vx_kms", "vy": "rel_vy_kms", "vz": "rel_vz_kms",
    }

    with st.expander("State configuration (optional)", expanded=False):
        st.write("Select relative position/velocity columns if they exist; otherwise demo state is used.")

        def pick_col(label, default_name):
            candidates = [c for c in cols if c.lower() == default_name.lower()]
            default = candidates[0] if candidates else None
            options = [None] + cols
            idx = 0 if default is None else (1 + cols.index(default))
            return st.selectbox(label, options=options, index=idx)

        cc1, cc2 = st.columns(2)
        with cc1:
            rx_col = pick_col("Relative X (km)", defaults["rx"])
            ry_col = pick_col("Relative Y (km)", defaults["ry"])
            rz_col = pick_col("Relative Z (km)", defaults["rz"])
        with cc2:
            vx_col = pick_col("Relative Vx (km/s)", defaults["vx"])
            vy_col = pick_col("Relative Vy (km/s)", defaults["vy"])
            vz_col = pick_col("Relative Vz (km/s)", defaults["vz"])

    r0_demo = np.array([0.3, 0.2, 0.05])
    v0_demo = np.array([-0.0006, 0.0004, 0.0])

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
    t = np.linspace(-span_hours * 3600, span_hours * 3600, N)

    r_nom = propagate_relative_motion(r0_km, v0_kms, t)
    dv_kms = delta_v / 1000.0
    v_man = apply_delta_v(v0_kms, dv_kms, burn_direction)
    r_man = propagate_relative_motion(r0_km, v_man, t)

    miss_nom_km = np.linalg.norm(r_nom, axis=1)
    miss_man_km = np.linalg.norm(r_man, axis=1)

    tca_idx = int(np.argmin(np.abs(t)))
    md_nom = float(miss_nom_km[tca_idx])
    md_man = float(miss_man_km[tca_idx])

    k1, k2, k3 = st.columns(3)
    k1.metric("Miss Distance @TCA (Nominal)", f"{md_nom * 1000:.1f} m")
    k2.metric("Miss Distance @TCA (Maneuvered)", f"{md_man * 1000:.1f} m")
    k3.metric("Increase", f"{(md_man - md_nom) * 1000:.1f} m")

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

# ==========================================================
# TAB 4: Reports
# ==========================================================
with tab_reports:
    st.subheader("Export Predictions")
    st.caption("Generate predicted risk for the whole dataset and download.")

    X_all_aligned, _ = build_model_input(df, model, feature_map, TARGET_COL)
    kind_all, values_all = get_risk_output(model, X_all_aligned)

    values_all = np.asarray(values_all, dtype=float).reshape(-1)
    risk_all = normalize_0_1(values_all)

    out = df.copy()
    out["predicted_raw"] = values_all
    out["predicted_risk_score_0_1"] = risk_all
    out["alert_flag"] = (risk_all >= risk_threshold).astype(int)

    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "⬇️ Download Predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="collision_risk_predictions.csv",
        mime="text/csv",
    )

# ==========================================================
# Conditional Footer (UPDATED)
#   Show "flat/constant" help ONLY when needed
# ==========================================================
st.divider()

# Compute flatness using the same arrays produced in Reports/Dataset
# (If something fails, we just skip the helper)
is_flat_footer = False
try:
    X_tmp, _ = build_model_input(df, model, feature_map, TARGET_COL)
    _, vals_tmp = get_risk_output(model, X_tmp)
    vals_tmp = np.asarray(vals_tmp, dtype=float).reshape(-1)
    is_flat_footer = (len(np.unique(vals_tmp)) <= 2) or (np.isclose(np.std(vals_tmp), 0.0))
except Exception:
    pass

if is_flat_footer:
    st.subheader("✅ If your risk chart is flat / constant")
    st.write(
        "If you see almost the same risk for every row, it usually means **feature mismatch**.\n\n"
        "**Best fix:** create a `feature_map.csv` in your repo like this:"
    )
    st.code(
        "model_feature,original_feature\n"
        "feature_0,time_to_tca\n"
        "feature_1,miss_distance\n"
        "feature_2,relative_speed\n"
        "...\n",
        language="text",
    )
    st.write("Your `feature_map.csv` must match the exact features and order used during training.")
else:
    st.success("✅ Risk outputs look non-constant. Feature mapping is working.")

if show_debug:
    st.info("Debug mode is ON (sidebar). Turn it OFF once everything is working.")

"""
requirements.txt (recommended)
----------------------------
streamlit
pandas
numpy
joblib
scikit-learn
plotly
"""
