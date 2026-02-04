import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Drury Dataset Explorer", layout="wide")
st.title("Drury Dataset Explorer")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    pq_path = os.path.join(DATA_DIR, "reports_main.parquet")
    st.write(f"Looking for: {pq_path}")
    st.write(f"File exists: {os.path.exists(pq_path)}")

    if os.path.exists(pq_path):
        st.write(f"File size: {os.path.getsize(pq_path)/1e6:.1f} MB")
        df = pd.read_parquet(pq_path)
        st.write(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
        st.write(f"Columns: {list(df.columns)}")
        st.write(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
        st.dataframe(df.head(20))
    else:
        files = os.listdir(DATA_DIR)
        st.write(f"Files in directory: {files}")
except Exception as e:
    st.error(f"Error: {type(e).__name__}: {e}")
