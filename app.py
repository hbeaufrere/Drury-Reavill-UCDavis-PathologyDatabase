import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Parquet files (preferred – smaller & faster)
PQ_FILES = {
    "Reports (Main)": os.path.join(DATA_DIR, "reports_main.parquet"),
    "Reports (Cytology file)": os.path.join(DATA_DIR, "cyto_reports.parquet"),
    "Cytology": os.path.join(DATA_DIR, "cyto_cytology.parquet"),
}

# Excel fallbacks
FILE_MAIN = os.path.join(DATA_DIR, "Database - Drury.xlsx")
FILE_CYTO = os.path.join(DATA_DIR, "Database Drury Cytology.xlsx")

# Max rows to display in tables (keeps browser responsive)
MAX_DISPLAY_ROWS = 5000

st.set_page_config(page_title="Drury Dataset Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_search_query(query_str):
    """
    Parse a human-friendly search string.
      - 'or'  between terms  -> match ANY term        (e.g. lymphoma or carcinoma)
      - 'and' between terms  -> match ALL terms        (e.g. lymphoma and liver)
      - comma ','            -> treated as OR
    Returns (mode, terms)  where mode is 'or' or 'and'.
    """
    raw = query_str.strip()
    if not raw:
        return ("or", [])

    has_and = bool(re.search(r'\band\b', raw, re.IGNORECASE))
    has_or  = bool(re.search(r'\bor\b',  raw, re.IGNORECASE))

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
    """Convert age text like '3 yr 2 mo' to a float in years."""
    if pd.isna(age_text):
        return None
    text = str(age_text).lower().strip()
    years = 0.0
    yr_match = re.search(r"(\d+)\s*(?:yr|year|years|y)", text)
    mo_match = re.search(r"(\d+)\s*(?:mo|month|months|m(?!i))", text)
    wk_match = re.search(r"(\d+)\s*(?:wk|week|weeks|w)", text)
    dy_match = re.search(r"(\d+)\s*(?:dy|day|days|d)", text)
    if yr_match:
        years += int(yr_match.group(1))
    if mo_match:
        years += int(mo_match.group(1)) / 12.0
    if wk_match:
        years += int(wk_match.group(1)) / 52.0
    if dy_match:
        years += int(dy_match.group(1)) / 365.0
    if years == 0.0:
        try:
            return float(text)
        except ValueError:
            return None
    return round(years, 2)


@st.cache_data(show_spinner="Loading data...")
def load_single_dataset(name):
    """Load a single dataset by name (lazy: only one at a time)."""
    pq_path = PQ_FILES.get(name)
    if pq_path and os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    elif name == "Reports (Main)" and os.path.exists(FILE_MAIN):
        df = pd.read_excel(FILE_MAIN, sheet_name="reports", engine="openpyxl")
    elif name == "Reports (Cytology file)" and os.path.exists(FILE_CYTO):
        df = pd.read_excel(FILE_CYTO, sheet_name="reports", engine="openpyxl")
    elif name == "Cytology" and os.path.exists(FILE_CYTO):
        df = pd.read_excel(FILE_CYTO, sheet_name="Cytology", engine="openpyxl")
    else:
        return None

    # Parse ages
    if "age_text" in df.columns:
        df["age_years"] = df["age_text"].apply(parse_age_to_years)
    elif "age" in df.columns:
        df["age_years"] = pd.to_numeric(df["age"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Sidebar - Dataset selector
# ---------------------------------------------------------------------------
st.sidebar.title("Drury Dataset Explorer")

# Determine available datasets
available = [name for name, path in PQ_FILES.items() if os.path.exists(path)]
if not available:
    # Try Excel fallbacks
    if os.path.exists(FILE_MAIN):
        available.append("Reports (Main)")
    if os.path.exists(FILE_CYTO):
        available.extend(["Reports (Cytology file)", "Cytology"])

if not available:
    st.error("No data files found. Place the data files in the same directory as this script.")
    st.stop()

dataset_name = st.sidebar.selectbox("Dataset", available)
df = load_single_dataset(dataset_name)

if df is None:
    st.error(f"Could not load dataset: {dataset_name}")
    st.stop()

st.sidebar.markdown(f"**{len(df):,}** rows | **{len(df.columns)}** columns")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_search, tab_graph, tab_raw = st.tabs(["Search", "Graphs", "Raw Data"])

# ========================== SEARCH TAB =====================================
with tab_search:
    st.header("Search & Filter")

    # --- free-text search ---
    searchable = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]
    search_cols = st.multiselect(
        "Columns to search in",
        options=searchable,
        default=[c for c in ["breed", "category", "diagnosis", "tissues", "specific_lesions"] if c in searchable],
    )
    query = st.text_input(
        "Search terms",
        help="Use **or** (or comma) between terms to match ANY: `lymphoma or carcinoma`\n\n"
             "Use **and** between terms to match ALL: `lymphoma and liver`\n\n"
             "A single phrase is searched as-is.",
    )

    # --- column filters ---
    st.subheader("Column Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "category" in df.columns:
            cats = sorted(df["category"].dropna().unique())
            sel_cats = st.multiselect("Category (species)", cats)
        else:
            sel_cats = []

    with col2:
        if "sex" in df.columns:
            sexes = sorted(df["sex"].dropna().unique())
            sel_sex = st.multiselect("Sex", sexes)
        else:
            sel_sex = []

    with col3:
        if "breed" in df.columns:
            breeds = sorted(df["breed"].dropna().unique())
            sel_breed = st.multiselect("Breed", breeds)
        else:
            sel_breed = []

    col4, col5 = st.columns(2)
    with col4:
        if "diagnosis_category" in df.columns:
            diag_cats = sorted(df["diagnosis_category"].dropna().unique())
            sel_diag = st.multiselect("Diagnosis category", diag_cats)
        else:
            sel_diag = []
    with col5:
        if "age_years" in df.columns and df["age_years"].notna().any():
            min_age = float(df["age_years"].min())
            max_age = float(df["age_years"].max())
            if max_age > min_age:
                age_range = st.slider("Age range (years)", min_age, max_age, (min_age, max_age))
            else:
                age_range = None
        else:
            age_range = None

    # Apply filters
    mask = pd.Series(True, index=df.index)

    if query and search_cols:
        mode, terms = parse_search_query(query)
        if terms:
            if mode == "or":
                text_mask = pd.Series(False, index=df.index)
                for term in terms:
                    for col in search_cols:
                        text_mask |= df[col].astype(str).str.contains(
                            term, case=False, na=False, regex=False
                        )
            else:
                text_mask = pd.Series(True, index=df.index)
                for term in terms:
                    term_found = pd.Series(False, index=df.index)
                    for col in search_cols:
                        term_found |= df[col].astype(str).str.contains(
                            term, case=False, na=False, regex=False
                        )
                    text_mask &= term_found
            mask &= text_mask

    if sel_cats:
        mask &= df["category"].isin(sel_cats)
    if sel_sex:
        mask &= df["sex"].isin(sel_sex)
    if sel_breed:
        mask &= df["breed"].isin(sel_breed)
    if sel_diag:
        mask &= df["diagnosis_category"].isin(sel_diag)
    if age_range and "age_years" in df.columns:
        mask &= df["age_years"].between(age_range[0], age_range[1]) | df["age_years"].isna()

    filtered = df.loc[mask]
    total_filtered = len(filtered)
    st.markdown(f"### Results: **{total_filtered:,}** / {len(df):,} rows")

    # Display columns selector
    default_display = [c for c in ["animal_name", "category", "breed", "sex", "age_text", "diagnosis",
                                    "diagnosis_category", "tissues", "specific_lesions"] if c in df.columns]
    show_cols = st.multiselect("Columns to display", options=df.columns.tolist(), default=default_display)

    display_df = filtered[show_cols] if show_cols else filtered
    if total_filtered > MAX_DISPLAY_ROWS:
        st.caption(f"Showing first {MAX_DISPLAY_ROWS:,} of {total_filtered:,} rows. Download CSV for full results.")
        display_df = display_df.head(MAX_DISPLAY_ROWS)
    st.dataframe(display_df, height=500, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered results as CSV", csv, "drury_filtered.csv", "text/csv")

# ========================== GRAPH TAB ======================================
with tab_graph:
    st.header("Graphs & Visualisations")

    gdf = filtered
    st.info(f"Graphing **{len(gdf):,}** rows (apply filters in the Search tab to narrow down)")

    graph_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Pie chart", "Histogram (numeric)", "Box plot",
         "Scatter plot", "Time series", "Heatmap (cross-tabulation)"],
    )

    if graph_type == "Bar chart":
        col_bar = st.selectbox("Count by column",
            [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 100], key="bar_col")
        top_n = st.slider("Show top N values", 5, 50, 20, key="bar_top")
        counts = gdf[col_bar].value_counts().head(top_n).reset_index()
        counts.columns = [col_bar, "count"]
        fig = px.bar(counts, x=col_bar, y="count", title=f"Top {top_n} - {col_bar}", text_auto=True)
        fig.update_layout(xaxis_tickangle=-45, height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Pie chart":
        col_pie = st.selectbox("Group by column",
            [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 100], key="pie_col")
        top_n_pie = st.slider("Show top N values", 5, 30, 10, key="pie_top")
        counts = gdf[col_pie].value_counts().head(top_n_pie).reset_index()
        counts.columns = [col_pie, "count"]
        fig = px.pie(counts, names=col_pie, values="count", title=f"Distribution - {col_pie}")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Histogram (numeric)":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        if not num_cols:
            st.warning("No numeric columns available.")
        else:
            col_hist = st.selectbox("Column", num_cols, key="hist_col")
            bins = st.slider("Number of bins", 10, 200, 50, key="hist_bins")
            color_by = st.selectbox("Color by (optional)",
                ["None"] + [c for c in gdf.columns if gdf[c].nunique() < 20], key="hist_color")
            color = None if color_by == "None" else color_by
            fig = px.histogram(gdf.dropna(subset=[col_hist]), x=col_hist, nbins=bins,
                               color=color, title=f"Histogram - {col_hist}")
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Box plot":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        cat_cols = [c for c in gdf.columns if gdf[c].dtype == "object" or gdf[c].nunique() < 50]
        if not num_cols or not cat_cols:
            st.warning("Need at least one numeric and one categorical column.")
        else:
            y_col = st.selectbox("Numeric column (Y axis)", num_cols, key="box_y")
            x_col = st.selectbox("Category column (X axis)", cat_cols, key="box_x")
            top_n_box = st.slider("Show top N categories", 5, 30, 10, key="box_top")
            top_cats = gdf[x_col].value_counts().head(top_n_box).index
            subset = gdf[gdf[x_col].isin(top_cats)].dropna(subset=[y_col])
            fig = px.box(subset, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            fig.update_layout(xaxis_tickangle=-45, height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Scatter plot":
        num_cols = [c for c in gdf.columns if pd.api.types.is_numeric_dtype(gdf[c])]
        if len(num_cols) < 2:
            st.warning("Need at least two numeric columns for a scatter plot.")
        else:
            x_sc = st.selectbox("X axis", num_cols, key="sc_x")
            y_sc = st.selectbox("Y axis", [c for c in num_cols if c != x_sc], key="sc_y")
            color_sc = st.selectbox("Color by (optional)",
                ["None"] + [c for c in gdf.columns if gdf[c].nunique() < 20], key="sc_color")
            color = None if color_sc == "None" else color_sc
            plot_data = gdf.dropna(subset=[x_sc, y_sc])
            if len(plot_data) > 5000:
                plot_data = plot_data.sample(n=5000, random_state=42)
            fig = px.scatter(plot_data, x=x_sc, y=y_sc, color=color,
                             title=f"{y_sc} vs {x_sc}", opacity=0.6)
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Time series":
        date_cols = [c for c in gdf.columns if "date" in c.lower()]
        if not date_cols:
            st.warning("No date columns detected.")
        else:
            date_col = st.selectbox("Date column", date_cols, key="ts_date")
            agg = st.selectbox("Aggregate by", ["Month", "Year", "Week"], key="ts_agg")
            temp = gdf[[date_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col])
            freq_map = {"Month": "ME", "Year": "YE", "Week": "W"}
            ts = temp.set_index(date_col).resample(freq_map[agg]).size().reset_index(name="count")
            fig = px.line(ts, x=date_col, y="count", title=f"Record count per {agg.lower()}")
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Heatmap (cross-tabulation)":
        cat_cols = [c for c in gdf.columns if gdf[c].dtype == "object" and 2 <= gdf[c].nunique() <= 50]
        if len(cat_cols) < 2:
            st.warning("Need at least two categorical columns with 2-50 unique values.")
        else:
            row_col = st.selectbox("Row variable", cat_cols, key="hm_row")
            col_col = st.selectbox("Column variable",
                [c for c in cat_cols if c != row_col], key="hm_col")
            ct = pd.crosstab(gdf[row_col], gdf[col_col])
            fig = px.imshow(ct, text_auto=True, title=f"{row_col} x {col_col}", aspect="auto")
            fig.update_layout(height=max(550, len(ct) * 22))
            st.plotly_chart(fig, use_container_width=True)

# ========================== RAW DATA TAB ===================================
with tab_raw:
    st.header("Raw Data")
    st.markdown(f"**Dataset:** {dataset_name} - {len(df):,} rows x {len(df.columns)} columns")

    st.subheader("Column overview")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [df[c].notna().sum() for c in df.columns],
        "Null": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(col_info, hide_index=True, use_container_width=True)

    st.subheader("Preview (first 100 rows)")
    st.dataframe(df.head(100), height=400, use_container_width=True)

    full_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full dataset as CSV", full_csv,
                        f"drury_{dataset_name.replace(' ', '_')}.csv", "text/csv")
