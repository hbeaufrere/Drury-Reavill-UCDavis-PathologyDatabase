import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Drury Dataset Explorer")
st.title("Drury Dataset Explorer - Test 2")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
pq_path = os.path.join(DATA_DIR, "reports_main.parquet")

st.write(f"File exists: {os.path.exists(pq_path)}")
st.write(f"File size: {os.path.getsize(pq_path)/1e6:.1f} MB")

df = pd.read_parquet(pq_path)
st.write(f"Loaded {len(df)} rows, {len(df.columns)} columns")
st.dataframe(df.head(10))
