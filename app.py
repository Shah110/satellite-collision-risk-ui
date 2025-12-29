import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Satellite Collision Risk UI", layout="centered")
st.title("🛰️ Satellite Collision Risk UI")
st.success("App is running ✅")

# Sidebar links
st.sidebar.header("🚀 Project Notebooks (Colab)")
st.sidebar.markdown(
    "- [Phase 1](https://colab.research.google.com/drive/1Utaq_FtgsHhV215frPX-CQhC6nMwkfs3?usp=sharing)\n"
    "- [Phase 2](https://colab.research.google.com/drive/1oog4BFnxr5ohss8HHJszAzv2rZqWFJO3?usp=sharing)\n"
    "- [Phase 3](https://colab.research.google.com/drive/1DGncV8CkSyTKbe5YBfqrTC-qrQxwfau-?usp=sharing)"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("combined_data.csv")

df = load_data()
st.subheader("📊 Loaded Dataset: combined_data.csv")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

# Load model
@st.cache_resource
def load_model():
    return joblib.load("collision_risk_model.pkl")

try:
    model = load_model()
    st.success("Model loaded ✅")
except Exception as e:
    st.error("Could not load collision_risk_model.pkl. Make sure it is uploaded to the repo root.")
    st.stop()

# --- Prediction ---
st.subheader("🧠 Run Prediction")

# If your combined_data.csv already has the target column, put its name here.
# If it does NOT have the target column, leave as "risk" and it will just ignore it.
TARGET_COL = "risk"

X = df.drop(columns=[TARGET_COL], errors="ignore")

if st.button("🚀 Predict Collision Risk"):
    try:
        preds = model.predict(X)
        out = df.copy()
        out["predicted_risk"] = preds

        st.success("Predictions generated ✅")
        st.dataframe(out.head())

        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            "collision_risk_predictions.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("This usually means the model expects different columns than combined_data.csv.")
