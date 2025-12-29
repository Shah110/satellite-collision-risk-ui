import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Satellite Collision Risk UI", layout="centered")
st.title("🛰️ Satellite Collision Risk UI")
st.success("App is running ✅")

# -----------------------------
# Sidebar: Links + Settings
# -----------------------------
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

st.sidebar.caption("Tip: If prediction fails, check missing/extra columns shown in the app.")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def align_features_for_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align input features to the model's expected feature names if available.
    - If model has feature_names_in_, reindex to those columns, fill missing with 0.
    - Otherwise, return X as-is.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        return X.reindex(columns=expected, fill_value=0)
    return X

# -----------------------------
# Load Dataset
# -----------------------------
st.header("📊 Dataset")

if not Path(DATA_FILE).exists():
    st.error(f"Dataset file not found: `{DATA_FILE}`. Upload it to the repo root or correct the name in sidebar.")
    st.stop()

df = load_csv(DATA_FILE)

col1, col2 = st.columns(2)
with col1:
    st.write("**File:**", DATA_FILE)
with col2:
    st.write("**Shape:**", df.shape)

with st.expander("Show dataset preview", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

st.caption("If you want to predict on different data, either replace the CSV in the repo or add an upload flow.")

# -----------------------------
# Load Model
# -----------------------------
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

# Show expected features if available
if hasattr(model, "feature_names_in_"):
    with st.expander("Model expected features", expanded=False):
        st.write(f"Model expects **{len(model.feature_names_in_)}** features.")
        st.write(list(model.feature_names_in_))

# -----------------------------
# Prediction Panel
# -----------------------------
st.header("🚀 Prediction Panel")

# Build X
X = df.drop(columns=[TARGET_COL], errors="ignore")

# Diagnostics: missing/extra columns (if model has expectations)
missing_cols, extra_cols = [], []
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    provided = list(X.columns)
    missing_cols = [c for c in expected if c not in provided]
    extra_cols = [c for c in provided if c not in expected]

diag1, diag2, diag3 = st.columns(3)
with diag1:
    st.metric("Provided features", len(X.columns))
with diag2:
    st.metric("Missing vs model", len(missing_cols))
with diag3:
    st.metric("Extra vs model", len(extra_cols))

if missing_cols:
    st.warning(f"Missing columns will be filled with 0 (showing up to 25): {missing_cols[:25]}")
if extra_cols:
    st.info(f"Extra columns will be ignored (showing up to 25): {extra_cols[:25]}")

# Align X to model if needed
X_aligned = align_features_for_model(model, X)

# Predict button
predict = st.button("🧠 Predict Collision Risk", type="primary")

if predict:
    try:
        preds = model.predict(X_aligned)
        out = df.copy()
        out["predicted_risk"] = preds

        st.success("Predictions generated ✅")

        # Show summary
        st.subheader("📈 Prediction Summary")
        st.write(out["predicted_risk"].describe())

        # Show results
        st.subheader("📄 Results Preview")
        st.dataframe(out.head(50), use_container_width=True)

        # Download
        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="collision_risk_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Most common cause: feature mismatch. "
            "If your model was trained with preprocessing, consider saving/loading a full pipeline."
        )
