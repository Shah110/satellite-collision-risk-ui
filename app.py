import streamlit as st

st.set_page_config(page_title="Satellite Collision Risk UI", layout="centered")
st.title("🛰️ Satellite Collision Risk UI")
st.write("App is running ✅")

st.sidebar.header("📌 Project Notebooks (Colab)")
st.sidebar.markdown("- [Phase 1](https://colab.research.google.com/drive/1Utaq_FtgsHhV215frPX-CQhC6nMwkfs3?usp=sharing)")
st.sidebar.markdown("- [Phase 2](https://colab.research.google.com/drive/1oog4BFnxr5ohss8HHJszAzv2rZqWFJO3?usp=sharing)")
st.sidebar.markdown("- [Phase 3](https://colab.research.google.com/drive/1DGncV8CkSyTKbe5YBfqrTC-qrQxwfau-?usp=sharing)")

st.info("Next: connect your trained model + data source.")
