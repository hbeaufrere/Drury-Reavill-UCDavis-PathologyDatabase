import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PQ_FILES = {
    "Reports (Main)": os.path.join(DATA_DIR, "reports_main.parquet"),
    "Reports (Cytology file)": os.path.join(DATA_DIR, "cyto_reports.parquet"),
    "Cytology": os.path.join(DATA_DIR, "cyto_cytology.parquet"),
}

MAX_DISPLAY_ROWS = 5000

st.set_page_config(page_title="Drury Dataset Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_search_query(query_str):
    raw = query_str.strip()
    if not raw:
        return ("or", [])
    has_and = bool(re.search(r'\band\b', raw, re.IGNORECASE))
    has_or = bool(re.search(r'\bor\b', raw, re.IGNORECASE))
    if has_and and not has_or:
        parts = re.split(r'\s+and\s+', raw, flags=re.IGNORECASE)
        return ("and", [p.strip() for p in parts if p.strip()])
    elif has_or and not has_and:
        parts = re.split(r'\s+or\s+', raw, flags=re.IGNORECASE)
        return ("or", [p.strip() for p in parts if p.strip()])
    elif has_and and has_or:
        parts = re.split(r'\s+and\s+', raw, flags=re.IGNORECASE)
        return ("and", [p.strip() for p in parts if p.strip()])
    elif ',' in raw:
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        return ("or", parts)
    else:
        return ("or", [raw])


def parse_age_to_years(age_text):
    if pd.isna(age_text):
        return None
    text = str(age_text).lower().strip()
    years = 0.0
    yr = re.search(r"(\d+)\s*(?:yr|year|years|y)", text)
    mo = re.search(r"(\d+)\s*(?:mo|month|months|m(?!i))", text)
    wk = re.search(r"(\d+)\s*(?:wk|week|weeks|w)", text)
    dy = re.search(r"(\d+)\s*(?:dy|day|days|d)", text)
    if yr: years += int(yr.group(1))
    if mo: years += int(mo.group(1)) / 12.0
    if wk: years += int(wk.group(1)) / 52.0
    if dy: years += int(dy.group(1)) / 365.0
    if years == 0.0:
        try:
            return float(text)
        except ValueError:
            return None
    return round(years, 2)


@st.cache_data(show_spinner="Loading data...")
def load_dataset(name):
    pq_path = PQ_FILES.get(name)
    if not pq_path or not os.path.exists(pq_path):
        return None
    df = pd.read_parquet(pq_path)
    if "age_text" in df.columns:
        df["age_years"] = df["age_text"].apply(parse_age_to_years)
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Drury Dataset Explorer")
available = [name for name, path in PQ_FILES.items() if os.path.exists(path)]
if not available:
    st.error("No data files found.")
    st.stop()

dataset_name = st.sidebar.selectbox("Dataset", available)
df = load_dataset(dataset_name)
if df is None:
    st.error("Could not load dataset.")
    st.stop()

st.sidebar.markdown(f"**{len(df):,}** rows | **{len(df.columns)}** columns")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_search, tab_graph, tab_raw = st.tabs(["Search", "Graphs", "Raw Data"])

# ========================== SEARCH TAB =====================================
with tab_search:
    st.header("Search & Filter")

    searchable = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]
    search_cols = st.multiselect(
        "Columns to search in", options=searchable,
        default=[c for c in ["breed", "category", "diagnosis", "tissues", "specific_lesions"] if c in searchable],
    )
    query = st.text_input(
        "Search terms",
        help="Use **or** (or comma) to match ANY term. Use **and** to match ALL terms.",
    )

    st.subheader("Column Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_cats = st.multiselect("Category (species)", sorted(df["category"].dropna().unique())) if "category" in df.columns else []
    with col2:
        sel_sex = st.multiselect("Sex", sorted(df["sex"].dropna().unique())) if "sex" in df.columns else []
    with col3:
        sel_breed = st.multiselect("Breed", sorted(df["breed"].dropna().unique())) if "breed" in df.columns else []

    col4, col5 = st.columns(2)
    with col4:
        sel_diag = st.multiselect("Diagnosis category", sorted(df["diagnosis_category"].dropna().unique())) if "diagnosis_category" in df.columns else []
    with col5:
        age_range = None
        if "age_years" in df.columns and df["age_years"].notna().any():
            mn, mx = float(df["age_years"].min()), float(df["age_years"].max())
            if mx > mn:
                age_range = st.slider("Age range (years)", mn, mx, (mn, mx))

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if query and search_cols:
        mode, terms = parse_search_query(query)
        if terms:
            if mode == "or":
                text_mask = pd.Series(False, index=df.index)
                for t in terms:
                    for c in search_cols:
                        text_mask |= df[c].astype(str).str.contains(t, case=False, na=False, regex=False)
            else:
                text_mask = pd.Series(True, index=df.index)
                for t in terms:
                    found = pd.Series(False, index=df.index)
                    for c in search_cols:
                        found |= df[c].astype(str).str.contains(t, case=False, na=False, regex=False)
                    text_mask &= found
            mask &= text_mask
    if sel_cats: mask &= df["category"].isin(sel_cats)
    if sel_sex: mask &= df["sex"].isin(sel_sex)
    if sel_breed: mask &= df["breed"].isin(sel_breed)
    if sel_diag: mask &= df["diagnosis_category"].isin(sel_diag)
    if age_range and "age_years" in df.columns:
        mask &= df["age_years"].between(age_range[0], age_range[1]) | df["age_years"].isna()

    filtered = df.loc[mask]
    total = len(filtered)
    st.markdown(f"### Results: **{total:,}** / {len(df):,} rows")

    default_cols = [c for c in ["animal_name", "category", "breed", "sex", "age_text", "diagnosis",
                                 "diagnosis_category", "tissues", "specific_lesions"] if c in df.columns]
    show_cols = st.multiselect("Columns to display", options=df.columns.tolist(), default=default_cols)

    display = filtered[show_cols] if show_cols else filtered
    if total > MAX_DISPLAY_ROWS:
        st.caption(f"Showing first {MAX_DISPLAY_ROWS:,} of {total:,} rows. Download CSV for full results.")
        display = display.head(MAX_DISPLAY_ROWS)
    st.dataframe(display, height=500, use_container_width=True)

    st.download_button("Download filtered results as CSV",
                        filtered.to_csv(index=False).encode("utf-8"),
                        "drury_filtered.csv", "text/csv")

# ========================== GRAPH TAB ======================================
with tab_graph:
    st.header("Graphs & Visualisations")
    gdf = filtered
    st.info(f"Graphing **{len(gdf):,}** rows (apply filters in Search tab to narrow down)")

    graph_type = st.selectbox("Chart type",
        ["Bar chart", "Pie chart", "Histogram (numeric)", "Box plot", "Scatter plot", "Heatmap (cross-tabulation)"])

    if graph_type == "Bar chart":
        col_bar = st.selectbox("Count by column",
            [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 100], key="bar_col")
        top_n = st.slider("Show top N", 5, 50, 20, key="bar_n")
        counts = gdf[col_bar].value_counts().head(top_n).reset_index()
        counts.columns = [col_bar, "count"]
        fig = px.bar(counts, x=col_bar, y="count", title=f"Top {top_n} - {col_bar}", text_auto=True)
        fig.update_layout(xaxis_tickangle=-45, height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Pie chart":
        col_pie = st.selectbox("Group by",
            [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 100], key="pie_col")
        top_n = st.slider("Show top N", 5, 30, 10, key="pie_n")
        counts = gdf[col_pie].value_counts().head(top_n).reset_index()
        counts.columns = [col_pie, "count"]
        fig = px.pie(counts, names=col_pie, values="count", title=f"Distribution - {col_pie}")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Histogram (numeric)":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            col_h = st.selectbox("Column", num_cols, key="hist_col")
            bins = st.slider("Bins", 10, 200, 50, key="hist_bins")
            fig = px.histogram(gdf.dropna(subset=[col_h]), x=col_h, nbins=bins, title=f"Histogram - {col_h}")
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Box plot":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        cat_cols = [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 50]
        if not num_cols or not cat_cols:
            st.warning("Need numeric and categorical columns.")
        else:
            y_col = st.selectbox("Numeric (Y)", num_cols, key="box_y")
            x_col = st.selectbox("Category (X)", cat_cols, key="box_x")
            top_n = st.slider("Top N categories", 5, 30, 10, key="box_n")
            top_cats = gdf[x_col].value_counts().head(top_n).index
            sub = gdf[gdf[x_col].isin(top_cats)].dropna(subset=[y_col])
            fig = px.box(sub, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            fig.update_layout(xaxis_tickangle=-45, height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Scatter plot":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns.")
        else:
            x_s = st.selectbox("X axis", num_cols, key="sc_x")
            y_s = st.selectbox("Y axis", [c for c in num_cols if c != x_s], key="sc_y")
            data = gdf.dropna(subset=[x_s, y_s])
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            fig = px.scatter(data, x=x_s, y=y_s, title=f"{y_s} vs {x_s}", opacity=0.6)
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Heatmap (cross-tabulation)":
        cat_cols = [c for c in gdf.columns if gdf[c].dtype == "object" and 2 <= gdf[c].nunique() <= 50]
        if len(cat_cols) < 2:
            st.warning("Need two categorical columns with 2-50 unique values.")
        else:
            r = st.selectbox("Row", cat_cols, key="hm_r")
            c = st.selectbox("Column", [x for x in cat_cols if x != r], key="hm_c")
            ct = pd.crosstab(gdf[r], gdf[c])
            fig = px.imshow(ct, text_auto=True, title=f"{r} x {c}", aspect="auto")
            fig.update_layout(height=max(550, len(ct) * 22))
            st.plotly_chart(fig, use_container_width=True)

# ========================== RAW DATA TAB ===================================
with tab_raw:
    st.header("Raw Data")
    st.markdown(f"**{dataset_name}** - {len(df):,} rows x {len(df.columns)} columns")

    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [df[c].notna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(col_info, hide_index=True, use_container_width=True)
    st.dataframe(df.head(100), height=400, use_container_width=True)

    st.download_button("Download full dataset as CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        f"drury_{dataset_name.replace(' ', '_')}.csv", "text/csv")
