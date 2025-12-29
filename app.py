import streamlit as st
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Satellite Collision Risk UI",
    layout="centered"
)

st.title("🛰️ Satellite Collision Risk UI")
st.success("App is running ✅")

# ---------------- Load CSV ----------------
@st.cache_data
def load_data():
    return pd.read_csv("combined_data.csv")

df = load_data()

st.subheader("📊 Loaded Dataset: combined_data.csv")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

# ---------------- Sidebar ----------------
st.sidebar.header("🚀 Project Notebooks (Colab)")
st.sidebar.markdown("- [Phase 1](https://colab.research.google.com/drive/1Utaq_FtgsHhV215frPX-CQhC6nMwkfs3?usp=sharing)")
st.sidebar.markdown("- [Phase 2](https://colab.research.google.com/drive/1oog4BFnxr5ohss8HHJszAz)
