import os
import urllib.request

# WFP Philippines Food Prices Dataset
# Source: https://data.humdata.org/dataset/wfp-food-prices-for-philippines
WFP_URL = (
    "https://data.humdata.org/dataset/ea251823-8694-47b4-82d0-7d27f00e8aba"
    "/resource/9a842d72-0d7d-4922-ad0e-eb8106c1ab0e"
    "/download/wfp_food_prices_phl.csv"
)

###
DESTINATION_FILE = os.path.join(os.path.dirname(__file__), "wfp_food_prices_phl.csv")

# Download
print("Downloading dataset from WFP / HDX...")
urllib.request.urlretrieve(WFP_URL, DESTINATION_FILE)

'''# Confirm
if os.path.exists(DESTINATION_FILE):
    size_mb = os.path.getsize(DESTINATION_FILE) / (1024 * 1024)
    print(f"✅ Dataset ready at: {DESTINATION_FILE} ({size_mb:.2f} MB)")
else:
    raise FileNotFoundError(
        "Download failed. Check your internet connection or\n"
        "verify the WFP URL is still active."
    )'''

###
if not os.path.exists(DESTINATION_FILE):
    print("Downloading dataset from WFP / HDX...")
    urllib.request.urlretrieve(WFP_URL, DESTINATION_FILE)
    
# IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
from datetime import datetime, date

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# PAGE CONFIG — Must be FIRST Streamlit call
st.set_page_config(
    page_title="Cavite Food Price Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS — Light yellow + gray-brown professional theme
st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #F7F8FC;
}

#MainMenu {display:none;}
footer {display:none;}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #021B4E 0%,
        #071A45 45%,
        #081733 100%
    );
}

/* REMOVE RADIO CIRCLES */
[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
}

[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Full-width buttons */
[data-testid="stSidebar"] .stRadio label {
    width: 100% !important;
    padding: 16px 18px !important;
    margin-bottom: 10px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Text */
[data-testid="stSidebar"] .stRadio label p {
    color: white !important;
    font-size: 15px;
    font-weight: 600;
}

/* Hover */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08);
    transform: translateX(3px);
}

/* Active menu */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    ) !important;

    border: none;

    box-shadow: 0 8px 20px rgba(37,99,235,.35);
}

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(74,55,40,0.10);
    border-left: 4px solid #2563EB;
    margin-bottom: 12px;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label {
    color: #64748B;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    color: #3A2A1A;
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.2;
}
.kpi-sub {
    color: #64748B;
    font-size: 0.75rem;
    margin-top: 4px;
}

/* Section Headers */
.section-header {
    background: linear-gradient(90deg, #2563EB,#3B82F6);
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 18px 0 14px 0;
    letter-spacing: 0.03em;
}

/* Insight Cards */
.insight-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 6px rgba(74,55,40,0.08);
    border-top: 3px solid #2563EB;
    margin-bottom: 10px;
}
.insight-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.insight-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #3A2A1A;
}

/* Prediction Result Box */
.prediction-box {
    background: linear-gradient(135deg, #EFF6FF,#DBEAFE);
    border: 2px solid #2563EB;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(196,151,62,0.15);
}
.prediction-price {
    font-size: 3rem;
    font-weight: 800;
    color: #4A3728;
    margin: 8px 0;
}
.prediction-label {
    font-size: 0.9rem;
    color: #64748B;
    font-weight: 500;
}

/* Alert Boxes */
.alert-increase {
    background: #FFF3E0;
    border-left: 4px solid #E65100;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #BF360C;
    font-size: 0.88rem;
}
.alert-stable {
    background: #E8F5E9;
    border-left: 4px solid #2E7D32;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #1B5E20;
    font-size: 0.88rem;
}
.alert-info {
    background: #E3F2FD;
    border-left: 4px solid #1565C0;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #0D47A1;
    font-size: 0.88rem;
}

/* Metric Badge */
.metric-good { color: #2E7D32; font-weight: 700; }
.metric-bad  { color: #C62828; font-weight: 700; }

/* Table styling */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Main content area */
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB,#3B82F6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8,#2563EB);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(74,55,40,0.25);
}

/* Selectbox / Slider labels */
label[data-testid="stWidgetLabel"] { color: #4A3728 !important; font-weight: 500; }



/* ── Fix light text on cream background ──────────────── */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #0F172A !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
}
.stMetric label {
    color: #5A4A3A !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #0F172A !important;
}
[data-testid="stMetricLabel"] {
    color: #5A4A3A !important;
}
/* Dataframe/table text */
.stDataFrame td, .stDataFrame th {
    color: #0F172A !important;
}
/* Selectbox, radio, slider */
.stSelectbox label, .stSlider label {
    color: #0F172A !important;
}

/* SIDEBAR TEXT FIX */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

[data-testid="stSidebar"] p {
    color: #D6E4FF !important;
}

</style>
""", unsafe_allow_html=True)

# PAGE 1 — DATA LOADING & PREPROCESSING
# Common file name variations the user might have
POSSIBLE_CSV_NAMES = [
    "wfp_food_prices_phl.csv",
    "wfp_food_prices_phl (1).csv",
    "philippines_food_prices.csv",
    "food_prices.csv",
    "data/wfp_food_prices_phl.csv",
    "/content/data/wfp_food_prices_phl.csv",
]

@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file=None):
    """
    Load the WFP Philippines food price CSV, filter for Cavite 2020–2026,
    clean it, and return a ready-to-use DataFrame.

    Column mapping (typical WFP CSV):
        date, admin1, admin2, market, category, commodity,
        commodity_id, unit, priceflag, pricetype, currency, price, usdprice
    """
    # 1. Load the raw CSV
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, low_memory=False)
    else:
        # Try known local paths
        found = False
        for path in POSSIBLE_CSV_NAMES:
            if os.path.exists(path):
                df_raw = pd.read_csv(path, low_memory=False)
                found = True
                break
        if not found:
            return None, "CSV_NOT_FOUND"

    # 2. Standardise column names (lowercase, strip spaces)
    df_raw.columns = df_raw.columns.str.lower().str.strip()

    # The WFP dataset sometimes uses 'admin2' for province
    # Cavite appears in admin2 (province) field
    cavite_col = None
    for col in ["admin2", "admin1"]:
        if col in df_raw.columns:
            cavite_col = col
            break
    if cavite_col is None:
        return None, "COLUMN_NOT_FOUND"

    # 3. Filter: Cavite, PHP, 2020–2026
    df = df_raw.copy()

    # Filter to Cavite
    df = df[df[cavite_col].str.contains("Cavite", case=False, na=False)]

    # Keep only PHP prices
    if "currency" in df.columns:
        df = df[df["currency"].str.upper() == "PHP"]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Filter years 2020–2026
    df = df[(df["year"] >= 2020) & (df["year"] <= 2026)]

    if df.empty:
        return None, "NO_CAVITE_DATA"



    # 4. Select & rename relevant columns
    rename_map = {}
    for src, dst in [("category", "category"), ("commodity", "commodity"),
                     ("commodity_id", "commodity_id"), ("price", "price"),
                     ("market", "market")]:
        if src in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    # Make sure we have the mandatory columns
    for col in ["commodity", "price", "year", "month", "category"]:
        if col not in df.columns:
            df[col] = "Unknown" if col in ["commodity", "category"] else 0

    # 5. Clean
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df = df.drop_duplicates()

    # Remove extreme outliers using IQR per commodity
    clean_frames = []
    for comm, grp in df.groupby("commodity"):
        q1, q3 = grp["price"].quantile(0.25), grp["price"].quantile(0.75)
        iqr = q3 - q1
        filtered = grp[(grp["price"] >= q1 - 3 * iqr) &
                       (grp["price"] <= q3 + 3 * iqr)]
        clean_frames.append(filtered)
    df = pd.concat(clean_frames).reset_index(drop=True)

    # 6. Encode features for ML
    # Commodity ID: use existing numeric ID or encode
    if "commodity_id" not in df.columns or df["commodity_id"].isna().all():
        le_comm = LabelEncoder()
        df["commodity_id"] = le_comm.fit_transform(df["commodity"].astype(str))
    else:
        df["commodity_id"] = pd.to_numeric(df["commodity_id"], errors="coerce")

        # ✅ Fix: wrap ndarray in Series with matching index before fillna
        missing_mask = df["commodity_id"].isna()
        if missing_mask.any():
            le_comm = LabelEncoder()
            encoded_array = le_comm.fit_transform(df["commodity"].astype(str))
            encoded_series = pd.Series(encoded_array, index=df.index)
            df["commodity_id"] = df["commodity_id"].fillna(encoded_series)

        df["commodity_id"] = df["commodity_id"].astype(int)

    # Category encoding: 0–3 based on category group
    cat_map = {}
    for i, cat in enumerate(sorted(df["category"].dropna().unique())):
        cat_map[cat] = i
    df["category_id"] = df["category"].map(cat_map).fillna(0).astype(int)

    # Month name for display
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df["month_name"] = df["month"].map(month_names)

    return df, "OK"


# PAGE 2 — SQLITE DATABASE
def init_database(df):
    """Create SQLite database and populate it from the cleaned DataFrame."""
    conn = sqlite3.connect("food_prices.db")
    cursor = conn.cursor()

    # Table: food_prices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            year INTEGER,
            month INTEGER,
            commodity TEXT,
            commodity_id INTEGER,
            category TEXT,
            price REAL
        )
    """)

    # Table: predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            commodity TEXT,
            prediction_month INTEGER,
            prediction_year INTEGER,
            predicted_price REAL,
            model_used TEXT,
            prediction_date TEXT
        )
    """)

    # Table: model_metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            model_name TEXT PRIMARY KEY,
            MAE REAL,
            RMSE REAL,
            R2 REAL
        )
    """)

    # Insert food price data (clear first to avoid duplicates on reload)
    cursor.execute("DELETE FROM food_prices")
    records = df[["date", "year", "month", "commodity", "commodity_id",
                  "category", "price"]].copy()
    records["date"] = records["date"].astype(str)
    records.to_sql("food_prices", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()


def save_prediction(commodity, month, year, predicted_price, model_used):
    """Save a prediction record to the database."""
    conn = sqlite3.connect("food_prices.db")
    conn.execute("""
        INSERT INTO predictions
        (commodity, prediction_month, prediction_year,
         predicted_price, model_used, prediction_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (commodity, month, year, predicted_price, model_used,
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


# PAGE — MACHINE LEARNING
@st.cache_data(show_spinner=False)
def train_models(df):
    """
    Train Linear Regression, Decision Tree, and Random Forest on the full
    Cavite dataset. Returns trained models, encoders, and evaluation metrics.

    Features: year, month, commodity_id, category_id
    Target  : price (₱)
    """
    # Feature matrix and target
    features = ["year", "month", "commodity_id", "category_id"]
    X = df[features].values
    y = df["price"].values

    # 80/20 train-test split, random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression":    LinearRegression(),
        "Decision Tree":        DecisionTreeRegressor(
                                    max_depth=8, random_state=42),
        "Random Forest":        RandomForestRegressor(
                                    n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1),
    }

    trained = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        trained[name] = model
        metrics[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # Persist metrics to database
    conn = sqlite3.connect("food_prices.db")
    conn.execute("DELETE FROM model_metrics")
    for name, m in metrics.items():
        conn.execute(
            "INSERT INTO model_metrics VALUES (?, ?, ?, ?)",
            (name, m["MAE"], m["RMSE"], m["R2"])
        )
    conn.commit()
    conn.close()

    return trained, metrics


def predict_price(trained_models, model_name, year, month,
                  commodity_id, category_id):
    """Generate a single price prediction."""
    model = trained_models[model_name]
    X_input = np.array([[year, month, commodity_id, category_id]])
    pred = model.predict(X_input)[0]
    return max(pred, 0.0)   # Price cannot be negative


def generate_forecast_series(trained_models, model_name,
                              commodity_id, category_id,
                              start_year, start_month, n_months):
    """Generate a sequence of monthly predictions."""
    results = []
    year, month = start_year, start_month
    for _ in range(n_months):
        pred = predict_price(trained_models, model_name,
                             year, month, commodity_id, category_id)
        results.append({
            "Year": year, "Month": month,
            "Date": f"{year}-{month:02d}",
            "Predicted Price (₱)": round(pred, 2)
        })
        month += 1
        if month > 12:
            month = 1
            year += 1
    return pd.DataFrame(results)



# PAGE 4 — PLOTLY CHART HELPERS (consistent color palette)
PALETTE = ["#2563EB","#3B82F6","#60A5FA","#93C5FD","#1D4ED8",
           "#06B6D4","#8B5CF6","#14B8A6"]


def make_line_chart(df_plot, x, y, color=None, title="", labels={}):
    fig = px.line(df_plot, x=x, y=y, color=color, title=title,
                  labels=labels, color_discrete_sequence=PALETTE)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#3A2A1A",
        title_font_size=14,
        title_font_color="#3A2A1A",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11,
                    font_color="#3A2A1A"),
        margin=dict(t=50, b=40, l=50, r=20),
        xaxis=dict(gridcolor="#E0D8C8", showline=True,
                   linecolor="#D4C4B0",
                   tickfont=dict(color="#3A2A1A", size=12),
                   title_font=dict(color="#3A2A1A")),
        yaxis=dict(gridcolor="#E0D8C8", showline=True,
                   linecolor="#D4C4B0",
                   tickfont=dict(color="#3A2A1A", size=12),
                   title_font=dict(color="#3A2A1A")),
    )
    return fig


def make_bar_chart(df_plot, x, y, color=None, title="", text=None):
    fig = px.bar(df_plot, x=x, y=y, color=color, title=title,
                 text=text, color_discrete_sequence=PALETTE)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#3A2A1A",
        title_font_size=14,
        title_font_color="#3A2A1A",
        margin=dict(t=50, b=40, l=50, r=20),
        xaxis=dict(gridcolor="#E0D8C8",
                   tickfont=dict(color="#3A2A1A", size=12),
                   title_font=dict(color="#3A2A1A")),
        yaxis=dict(gridcolor="#E0D8C8",
                   tickfont=dict(color="#3A2A1A", size=12),
                   title_font=dict(color="#3A2A1A")),
    )
    fig.update_traces(textposition="outside",
                      textfont=dict(color="#3A2A1A"))
    return fig


# PAGE 5 — SIDEBAR NAVIGATION
# ===== SIDEBAR =====

with st.sidebar:

    # Logo
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2920/2920349.png",
        width=60
    )

    # Title
    st.markdown("""
<div style="
    color:white;
    font-size:32px;
    font-weight:700;
    line-height:1.2;
    margin-top:10px;
">
    Cavite Food Price<br>
    Forecasting System
</div>

<div style="
    color:#D6E4FF;
    font-size:14px;
    margin-top:10px;
">
    ML-Based Price Prediction & Analysis
</div>
""", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "",
        [
            "Dashboard",
            "Food Price Trends",
            "Food Price Forecasting",
            "Commodity Explorer",
            "Model Performance",
            "Decision Support Insights",
            "About the Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style='
        color:#B8C7E0;
        font-size:12px;
        line-height:1.8;
    '>
        <b>Food Price Analytics</b><br>
        Capstone Project 2025<br>
        College of Engineering
    </div>
    """, unsafe_allow_html=True)


# SECTION 6 — LOAD DATA & TRAIN MODELS
with st.spinner("Loading and preprocessing data…"):
    df, status = load_and_preprocess(None)

if status != "OK" or df is None:
    st.error("Dataset not found or could not be loaded.")
    st.markdown("""
    **How to fix this:**
    1. Upload your `wfp_food_prices_phl.csv` using the sidebar uploader, **OR**
    2. Place the file in the same folder as `app.py` before running.

    **Download the dataset from (official WFP/HDX source):**
    https://data.humdata.org/dataset/wfp-food-prices-for-philippines

    **Error detail:** {} — {}
    """.format(status, {
        "CSV_NOT_FOUND": "CSV file not found in any expected location.",
        "COLUMN_NOT_FOUND": "Could not find admin1/admin2 column for Cavite filtering.",
        "NO_CAVITE_DATA": "No records found for Cavite in the 2020–2026 range.",
    }.get(status, "Unknown error.")))
    st.stop()


# Initialise database only once per session
if "db_initialized" not in st.session_state:
    init_database(df)
    st.session_state["db_initialized"] = True

# Train models
with st.spinner("Training machine learning models…"):
    trained_models, model_metrics = train_models(df)

# Derived lookup structures
commodities      = sorted(df["commodity"].unique())
categories       = sorted(df["category"].dropna().unique())
commodity_to_id  = df.groupby("commodity")["commodity_id"].first().to_dict()
commodity_to_cat = df.groupby("commodity")["category"].first().to_dict()
cat_to_id        = df.groupby("category")["category_id"].first().to_dict()
best_model_name  = min(model_metrics, key=lambda n: model_metrics[n]["MAE"])
month_names_full = {1:"January",2:"February",3:"March",4:"April",
                    5:"May",6:"June",7:"July",8:"August",
                    9:"September",10:"October",11:"November",12:"December"}



# SECTION 1 — DASHBOARD


if "Dashboard" in page:
    st.markdown("##  Dashboard")
    st.markdown("*Cavite Food Price Analytics — Monthly Overview 2020–2026*")

    # KPI Cards
    latest_date  = df["date"].max()
    avg_price    = df["price"].mean()
    total_recs   = len(df)
    total_comms  = df["commodity"].nunique()
    max_comm     = df.groupby("commodity")["price"].mean().idxmax()
    min_comm     = df.groupby("commodity")["price"].mean().idxmin()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Total Commodities", total_comms, "tracked in Cavite"),
        (k2, "Total Records",     f"{total_recs:,}", "cleaned entries"),
        (k3, "Average Price",     f"₱{avg_price:,.2f}", "all commodities"),
        (k4, "Highest-Priced",    max_comm[:18],  "by avg price"),
        (k5, "Lowest-Priced",     min_comm[:18],  "by avg price"),
        (k6, "Latest Data",       latest_date.strftime("%b %Y"), "most recent entry"),
    ]
    for col, label, val, sub in kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: Monthly average + Historical trend
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="section-header"> Monthly Average Food Prices</div>',
                    unsafe_allow_html=True)
        monthly_avg = (df.groupby(["year", "month", "month_name"])["price"]
                       .mean().reset_index())
        monthly_avg["period"] = (monthly_avg["year"].astype(str) + "-"
                                 + monthly_avg["month"].astype(str).str.zfill(2))
        monthly_avg = monthly_avg.sort_values("period")
        fig = make_line_chart(monthly_avg, "period", "price",
                              title="Monthly Average Price (All Commodities)",
                              labels={"price": "Avg Price (₱)", "period": "Month"})
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header"> Price Distribution by Category</div>',
                    unsafe_allow_html=True)
        cat_avg = df.groupby("category")["price"].mean().reset_index()
        fig2 = px.pie(cat_avg, names="category", values="price",
                      color_discrete_sequence=PALETTE,
                      title="Share of Average Price by Category")
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           font_family="Inter", font_color="#3A2A1A",
                           title_font_size=14, title_font_color="#3A2A1A",
                           legend=dict(font=dict(color="#3A2A1A", size=12)))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Top 10 + Year-over-year
    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown('<div class="section-header"> Top 10 Most Expensive Commodities</div>',
                    unsafe_allow_html=True)
        top10 = (df.groupby("commodity")["price"].mean()
                   .sort_values(ascending=False).head(10).reset_index())
        top10.columns = ["Commodity", "Average Price (₱)"]
        top10["Average Price (₱)"] = top10["Average Price (₱)"].round(2)
        fig3 = make_bar_chart(top10, "Average Price (₱)", "Commodity",
                              title="Top 10 Commodities by Average Price",
                              text="Average Price (₱)")
        fig3.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header"> Yearly Average Price Trend</div>',
                    unsafe_allow_html=True)
        yr_avg = df.groupby("year")["price"].mean().reset_index()
        fig4 = make_bar_chart(yr_avg, "year", "price",
                              title="Average Price per Year",
                              text="price")
        fig4.update_traces(text=yr_avg["price"].round(0).astype(int).astype(str).add(" ₱"))
        st.plotly_chart(fig4, use_container_width=True)

    # Insights
    st.markdown('<div class="section-header"> Quick Insights</div>',
                unsafe_allow_html=True)

    growth = {}
    for comm, grp in df.groupby("commodity"):
        grp = grp.sort_values(["year", "month"])
        if len(grp) >= 2:
            growth[comm] = ((grp["price"].iloc[-1] - grp["price"].iloc[0])
                            / grp["price"].iloc[0] * 100)

    growth_series = pd.Series(growth)
    highest_growth = growth_series.idxmax()
    highest_growth_val = growth_series.max()

    volatility = df.groupby("commodity")["price"].std()
    most_volatile = volatility.idxmax()
    most_stable   = volatility.idxmin()

    ic1, ic2, ic3, ic4 = st.columns(4)
    insights = [
        (ic1, "Highest Growth Commodity",
              f"{highest_growth}", f"+{highest_growth_val:.1f}% over period"),
        (ic2, "Most Expensive Commodity",
              max_comm, f"₱{df[df['commodity']==max_comm]['price'].mean():.2f} avg"),
        (ic3, "Highest Price Volatility",
              most_volatile, "Largest std deviation"),
        (ic4, "Most Stable Commodity",
              most_stable, "Lowest std deviation"),
    ]
    for col, title, val, sub in insights:
        col.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# SECTION 2 — FOOD PRICE TRENDS

elif "Trends" in page:
    st.markdown("##  Food Price Trends")
    st.markdown("*Explore historical price movements for any commodity or category*")

    # Filters
    with st.expander(" Filter Options", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_comm_trend = st.selectbox("Commodity", ["All"] + commodities)
        with fc2:
            sel_cat_trend  = st.selectbox("Category", ["All"] + categories)
        with fc3:
            yr_range = st.slider("Year Range", 2020, 2026, (2020, 2026))

    # Apply filters
    dft = df.copy()
    if sel_comm_trend != "All":
        dft = dft[dft["commodity"] == sel_comm_trend]
    if sel_cat_trend != "All":
        dft = dft[dft["category"] == sel_cat_trend]
    dft = dft[(dft["year"] >= yr_range[0]) & (dft["year"] <= yr_range[1])]

    if dft.empty:
        st.warning("No data for the selected filters. Please adjust.")
        st.stop()

    # Summary Stats
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "Average Price",   f"₱{dft['price'].mean():.2f}"),
        (s2, "Minimum Price",   f"₱{dft['price'].min():.2f}"),
        (s3, "Maximum Price",   f"₱{dft['price'].max():.2f}"),
        (s4, "Price Std Dev",   f"₱{dft['price'].std():.2f}"),
    ]
    for col, lbl, val in stats:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    # Historical Trend Chart
    st.markdown('<div class="section-header"> Historical Price Trend</div>',
                unsafe_allow_html=True)
    trend_data = (dft.groupby(["year", "month"])["price"].mean().reset_index())
    trend_data["period"] = (trend_data["year"].astype(str) + "-"
                            + trend_data["month"].astype(str).str.zfill(2))
    trend_data = trend_data.sort_values("period")

    fig_trend = make_line_chart(trend_data, "period", "price",
                                title="Historical Average Price Over Time",
                                labels={"price": "Avg Price (₱)", "period": "Month"})
    fig_trend.add_scatter(x=trend_data["period"], y=trend_data["price"],
                          mode="markers", marker=dict(size=5, color="#1D4ED8"),
                          name="Data Points", showlegend=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Monthly Movement + YoY
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown('<div class="section-header"> Monthly Price Movement</div>',
                    unsafe_allow_html=True)
        monthly_box = dft.groupby(["month", "month_name"])["price"].mean().reset_index()
        fig_box = px.bar(monthly_box, x="month_name", y="price",
                         color="price", color_continuous_scale=["#93C5FD", "#1D4ED8"],
                         title="Average Price by Month",
                         labels={"price": "Avg Price (₱)", "month_name": "Month"})
        fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_family="Inter", font_color="#3A2A1A",
                              title_font_size=13, title_font_color="#3A2A1A",
                              xaxis=dict(tickfont=dict(color="#3A2A1A"),
                                         title_font=dict(color="#3A2A1A")),
                              yaxis=dict(tickfont=dict(color="#3A2A1A"),
                                         title_font=dict(color="#3A2A1A")),
                              coloraxis_colorbar=dict(
                                  tickfont=dict(color="#3A2A1A"),
                                  title_font=dict(color="#3A2A1A")))
        st.plotly_chart(fig_box, use_container_width=True)

    with c_b:
        st.markdown('<div class="section-header"> Year-over-Year Comparison</div>',
                    unsafe_allow_html=True)
        yoy = dft.groupby("year")["price"].mean().reset_index()
        fig_yoy = make_bar_chart(yoy, "year", "price",
                                 title="Average Price per Year (YoY Comparison)",
                                 text="price")
        fig_yoy.update_traces(
            text=yoy["price"].round(0).astype(int).astype(str).add(" ₱")
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    # Seasonal Analysis
    st.markdown('<div class="section-header"> Seasonal Pattern (Average by Month across Years)</div>',
                unsafe_allow_html=True)
    seasonal = dft.groupby(["year", "month", "month_name"])["price"].mean().reset_index()
    fig_sea = px.line(seasonal, x="month", y="price", color="year",
                      color_discrete_sequence=PALETTE,
                      title="Seasonal Trend — Monthly Average per Year",
                      labels={"price": "Avg Price (₱)", "month": "Month"})
    fig_sea.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Inter", font_color="#3A2A1A",
                          title_font_size=13, title_font_color="#3A2A1A",
                          legend=dict(font=dict(color="#3A2A1A", size=11)),
                          xaxis=dict(tickvals=list(range(1, 13)),
                                     ticktext=list(month_names_full.values()),
                                     tickfont=dict(color="#3A2A1A"),
                                     title_font=dict(color="#3A2A1A")),
                          yaxis=dict(tickfont=dict(color="#3A2A1A"),
                                     title_font=dict(color="#3A2A1A")))
    st.plotly_chart(fig_sea, use_container_width=True)


# SECTION 3 — FOOD PRICE FORECASTING

elif "Forecast" in page:
    st.markdown("##  Food Price Forecasting")
    st.markdown("*Predict future food prices using machine learning regression models*")

    # Input Panel
    st.markdown('<div class="section-header"> Prediction Parameters</div>',
                unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        sel_comm_fc = st.selectbox(" Commodity", commodities)
    with p2:
        sel_month_fc = st.selectbox(" Prediction Month",
                                    list(month_names_full.values()))
        sel_month_num = {v: k for k, v in month_names_full.items()}[sel_month_fc]
    with p3:
        sel_year_fc = st.selectbox(" Prediction Year",
                                   list(range(2024, 2030)),
                                   index=1)

    p4, p5 = st.columns(2)
    with p4:
        sel_model_fc = st.selectbox(" ML Model",
                                    list(trained_models.keys()),
                                    index=list(trained_models.keys()).index(best_model_name))
    with p5:
        horizon = st.selectbox(" Forecast Horizon",
                               ["Single Month", "Next 3 Months",
                                "Next 6 Months", "Next 12 Months"])

    st.markdown("")
    run_pred = st.button(" Generate Prediction")

    if run_pred:
        comm_id  = commodity_to_id.get(sel_comm_fc, 0)
        cat_name = commodity_to_cat.get(sel_comm_fc, categories[0])
        cat_id   = cat_to_id.get(cat_name, 0)

        horizon_map = {"Single Month": 1, "Next 3 Months": 3,
                       "Next 6 Months": 6, "Next 12 Months": 12}
        n_months = horizon_map[horizon]

        forecast_df = generate_forecast_series(
            trained_models, sel_model_fc,
            comm_id, cat_id,
            sel_year_fc, sel_month_num, n_months
        )

        # Primary result box
        first_pred = forecast_df["Predicted Price (₱)"].iloc[0]
        save_prediction(sel_comm_fc, sel_month_num, sel_year_fc,
                        first_pred, sel_model_fc)

        col_res, col_inf = st.columns([1, 1.4])
        with col_res:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Predicted Price for<br>
                    <b>{sel_comm_fc}</b><br>
                    {sel_month_fc} {sel_year_fc}
                </div>
                <div class="prediction-price">₱{first_pred:,.2f}</div>
                <div class="prediction-label">
                    Model: <b>{sel_model_fc}</b><br>
                    Category: <b>{cat_name}</b>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_inf:
            # Show model accuracy for context
            m = model_metrics[sel_model_fc]
            st.markdown("**Model Performance (on held-out test set)**")
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | MAE    | ₱{m['MAE']:.2f} |
            | RMSE   | ₱{m['RMSE']:.2f} |
            | R      | {m['R2']:.4f} |
            """)
            st.info(f"The model's average prediction error is approximately "
                    f"**₱{m['MAE']:.2f}**, so the real price is likely between "
                    f"**₱{max(0, first_pred - m['MAE']):.2f}** and "
                    f"**₱{first_pred + m['MAE']:.2f}**.")

        # Forecast table + chart
        if n_months > 1:
            st.markdown('<div class="section-header"> Forecast Series</div>',
                        unsafe_allow_html=True)

            # Get historical data for this commodity
            hist = (df[df["commodity"] == sel_comm_fc]
                    .groupby(["year", "month"])["price"].mean().reset_index())
            hist["Date"] = (hist["year"].astype(str) + "-"
                            + hist["month"].astype(str).str.zfill(2))
            hist = hist.sort_values("Date")

            fig_fc = go.Figure()
            # Historical line
            fig_fc.add_trace(go.Scatter(
                x=hist["Date"], y=hist["price"],
                name="Historical Price", mode="lines+markers",
                line=dict(color="#4A3728", width=2),
                marker=dict(size=4)
            ))
            # Forecast line
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Predicted Price (₱)"],
                name=f"Forecast ({sel_model_fc})", mode="lines+markers",
                line=dict(color="#C4973E", width=2.5, dash="dot"),
                marker=dict(size=7, symbol="diamond")
            ))



            fig_fc.update_layout(
                title=f"{sel_comm_fc} — Historical vs Forecast",
                plot_bgcolor="white", paper_bgcolor="white",
                font_family="Inter", font_color="#3A2A1A",
                title_font_size=14, title_font_color="#3A2A1A",
                xaxis_title="Month", yaxis_title="Price (₱)",
                xaxis=dict(tickfont=dict(color="#3A2A1A"),
                          title_font=dict(color="#3A2A1A")),
                yaxis=dict(tickfont=dict(color="#3A2A1A"),
                          title_font=dict(color="#3A2A1A")),
                legend=dict(orientation="h", y=-0.2,
                            font=dict(color="#3A2A1A", size=11)),
                margin=dict(t=50, b=80, l=50, r=20)
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast table
            st.markdown('<div class="section-header">📋 Forecast Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                forecast_df.style.format({"Predicted Price (₱)": "₱{:.2f}"}),
                use_container_width=True
            )




# SECTION 4 — COMMODITY EXPLORER

elif "Explorer" in page:
    st.markdown("##  Commodity Explorer")
    st.markdown("*Deep-dive analytical profile for any individual commodity*")

    sel_comm_ex = st.selectbox("Select Commodity", commodities)
    df_ex = df[df["commodity"] == sel_comm_ex].copy()

    cat_ex = commodity_to_cat.get(sel_comm_ex, "N/A")
    st.caption(f"**Category:** {cat_ex}")

    # Stats row
    ea, eb, ec, ed, ee = st.columns(5)
    stats_ex = [
        (ea, "Average Price",  f"₱{df_ex['price'].mean():.2f}"),
        (eb, "Lowest Price",   f"₱{df_ex['price'].min():.2f}"),
        (ec, "Highest Price",  f"₱{df_ex['price'].max():.2f}"),
        (ed, "Std Deviation",  f"₱{df_ex['price'].std():.2f}"),
        (ee, "Records",        f"{len(df_ex):,}"),
    ]
    for col, lbl, val in stats_ex:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{lbl}</div>
            <div class="kpi-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    # Full historical trend
    st.markdown('<div class="section-header"> Full Historical Price Trend</div>',
                unsafe_allow_html=True)
    ex_trend = (df_ex.groupby(["year", "month"])["price"].mean().reset_index())
    ex_trend["period"] = (ex_trend["year"].astype(str) + "-"
                          + ex_trend["month"].astype(str).str.zfill(2))
    ex_trend = ex_trend.sort_values("period")

    fig_ex1 = go.Figure()
    fig_ex1.add_trace(go.Scatter(
        x=ex_trend["period"], y=ex_trend["price"],
        mode="lines+markers", fill="tozeroy",
        fillcolor="rgba(196,151,62,0.12)",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=5)
    ))
    fig_ex1.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", font_color="#3A2A1A",
        title=f"{sel_comm_ex} — Monthly Average Price (2020–2026)",
        title_font_color="#3A2A1A",
        xaxis_title="Month", yaxis_title="Price (₱)",
        xaxis=dict(tickfont=dict(color="#3A2A1A"),
                   title_font=dict(color="#3A2A1A")),
        yaxis=dict(tickfont=dict(color="#3A2A1A"),
                   title_font=dict(color="#3A2A1A")),
        margin=dict(t=50, b=40, l=50, r=20)
    )
    st.plotly_chart(fig_ex1, use_container_width=True)

    # Seasonal heatmap
    ex_heat = (df_ex.groupby(["year", "month"])["price"].mean().reset_index()
               .pivot(index="year", columns="month", values="price"))
    ex_heat.columns = [month_names_full[m] for m in ex_heat.columns]

    fig_heat = px.imshow(ex_heat,
                         color_continuous_scale=["#EFF6FF", "#3B82F6", "#1D4ED8"],
                         title=f"{sel_comm_ex} — Price Heatmap (Year × Month)",
                         labels=dict(color="Price (₱)"))
    fig_heat.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           font_family="Inter", font_color="#3A2A1A",
                           title_font_size=13, title_font_color="#3A2A1A",
                           xaxis=dict(tickfont=dict(color="#3A2A1A")),
                           yaxis=dict(tickfont=dict(color="#3A2A1A")),
                           coloraxis_colorbar=dict(
                               tickfont=dict(color="#3A2A1A"),
                               title_font=dict(color="#3A2A1A")))
    st.plotly_chart(fig_heat, use_container_width=True)
    # Quick forecast for this commodity
    st.markdown('<div class="section-header"> Quick Forecast (Next 6 Months)</div>',
                unsafe_allow_html=True)

    comm_id_ex = commodity_to_id.get(sel_comm_ex, 0)
    cat_id_ex  = cat_to_id.get(cat_ex, 0)
    today = date.today()

    fc6 = generate_forecast_series(
        trained_models, best_model_name,
        comm_id_ex, cat_id_ex,
        today.year, today.month, 6
    )
    st.dataframe(
        fc6.style.format({"Predicted Price (₱)": "₱{:.2f}"}),
        use_container_width=True
    )
    st.caption(f"Using best model: **{best_model_name}**")


# SECTION 5 — MODEL PERFORMANCE

elif "Performance" in page:
    st.markdown("##  Model Performance")
    st.markdown("*Compare regression model accuracy on the Cavite food price dataset*")

    metrics_df = pd.DataFrame(model_metrics).T.reset_index()
    metrics_df.columns = ["Model", "MAE (₱)", "RMSE (₱)", "R Score"]

    # Performance Table
    st.markdown('<div class="section-header"> Model Comparison Table</div>',
                unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
            .format({"MAE (₱)": "₱{:.4f}", "RMSE (₱)": "₱{:.4f}",
                     "R Score": "{:.4f}"})
            .highlight_min(subset=["MAE (₱)", "RMSE (₱)"],
                           color="#D4EDDA")
            .highlight_max(subset=["R Score"],
                           color="#D4EDDA"),
        use_container_width=True
    )
    st.caption("Green = best performance. Lower MAE/RMSE is better. "
               "Higher R (closer to 1.0) is better.")

    # Best model callout
    bm = model_metrics[best_model_name]
    st.success(f"**Best Model: {best_model_name}** — "
               f"MAE=₱{bm['MAE']:.2f}, RMSE=₱{bm['RMSE']:.2f}, R={bm['R2']:.4f}")

    # Grouped bar chart
    st.markdown('<div class="section-header"> Performance Comparison Charts</div>',
                unsafe_allow_html=True)

    m_long = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")


    for metric in ["MAE (₱)", "RMSE (₱)", "R Score"]:
      subset = m_long[m_long["Metric"] == metric].reset_index(drop=True)  # ← reset index
      fig_m  = px.bar(subset, x="Model", y="Value", color="Model",
                      color_discrete_sequence=PALETTE,
                      title=f"{metric} by Model",
                      labels={"Value": metric},
                      text="Value")  # ← let px.bar handle the text directly
      fig_m.update_traces(
          texttemplate="%{text:.4f}",  # ← format it here instead
          textposition="outside",
          textfont=dict(color="#3A2A1A")
      )
      fig_m.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          font_family="Inter", font_color="#3A2A1A",
                          showlegend=False,
                          title_font_size=13, title_font_color="#3A2A1A",
                          xaxis=dict(tickfont=dict(color="#3A2A1A"),
                                      title_font=dict(color="#3A2A1A")),
                          yaxis=dict(tickfont=dict(color="#3A2A1A"),
                                      title_font=dict(color="#3A2A1A")),
                          margin=dict(t=45, b=30))
      st.plotly_chart(fig_m, use_container_width=True)



    # Prediction vs actual scatter
    st.markdown('<div class="section-header"> Predicted vs Actual (Test Set Sample)</div>',
                unsafe_allow_html=True)

    # Re-compute test-set predictions for visualisation
    features = ["year", "month", "commodity_id", "category_id"]
    X  = df[features].values
    y  = df["price"].values
    _, X_test_v, _, y_test_v = train_test_split(X, y, test_size=0.2, random_state=42)

    scatter_cols = st.columns(len(trained_models))
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred_v = model.predict(X_test_v)
        sample_n = min(300, len(y_test_v))
        np.random.seed(42)
        idx      = np.random.choice(len(y_test_v), sample_n, replace=False)

        fig_sc = px.scatter(
            x=y_test_v[idx], y=y_pred_v[idx],
            labels={"x": "Actual Price (₱)", "y": "Predicted Price (₱)"},
            title=name, color_discrete_sequence=[PALETTE[i]]
        )
        # Perfect prediction line
        mn = min(y_test_v[idx].min(), y_pred_v[idx].min())
        mx = max(y_test_v[idx].max(), y_pred_v[idx].max())
        fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                         line=dict(color="#C62828", dash="dash", width=1.5))
        fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_family="Inter", font_color="#3A2A1A",
                              title_font_size=12, title_font_color="#3A2A1A",
                              xaxis=dict(tickfont=dict(color="#3A2A1A"),
                                         title_font=dict(color="#3A2A1A")),
                              yaxis=dict(tickfont=dict(color="#3A2A1A"),
                                         title_font=dict(color="#3A2A1A")),
                              margin=dict(t=40, b=30, l=40, r=10))
        scatter_cols[i].plotly_chart(fig_sc, use_container_width=True)

    st.caption("Red dashed line = perfect prediction. "
               "Points close to the line = accurate model.")


# SECTION 6 — DECISION SUPPORT INSIGHTS

elif "Insights" in page:
    st.markdown("##  Decision Support Insights")
    st.markdown("*Data-driven recommendations to help consumers plan smarter purchases*")

    today = date.today()
    curr_year, curr_month = today.year, today.month

    # Generate next-month predictions for ALL commodities
    next_month = curr_month + 1 if curr_month < 12 else 1
    next_year  = curr_year  if curr_month < 12 else curr_year + 1

    predictions_now  = {}
    predictions_next = {}

    for comm in commodities:
        cid   = commodity_to_id.get(comm, 0)
        cat   = commodity_to_cat.get(comm, categories[0])
        catid = cat_to_id.get(cat, 0)

        predictions_now[comm]  = predict_price(
            trained_models, best_model_name,
            curr_year, curr_month, cid, catid
        )
        predictions_next[comm] = predict_price(
            trained_models, best_model_name,
            next_year, next_month, cid, catid
        )

    # Calculate price change %
    changes = {
        comm: ((predictions_next[comm] - predictions_now[comm])
               / predictions_now[comm] * 100)
        if predictions_now[comm] > 0 else 0
        for comm in commodities
    }

    # Consumer Alerts
    st.markdown('<div class="section-header"> Monthly Consumer Price Alerts</div>',
                unsafe_allow_html=True)
    st.markdown(f"**Forecast Period:** {month_names_full[next_month]} {next_year} "
                f"*(using {best_model_name})*")

    increasing = [(c, v) for c, v in changes.items() if v > 3]
    stable     = [(c, v) for c, v in changes.items() if -3 <= v <= 3]
    decreasing = [(c, v) for c, v in changes.items() if v < -3]

    increasing.sort(key=lambda x: x[1], reverse=True)
    stable.sort(key=lambda x: abs(x[1]))

    al1, al2 = st.columns(2)
    with al1:
        st.markdown("**Price Increase Expected**")
        if increasing:
            for comm, chg in increasing[:8]:
                st.markdown(
                    f'<div class="alert-increase"> <b>{comm}</b> '
                    f'— Expected +{chg:.1f}% '
                    f'(₱{predictions_next[comm]:.2f})</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown('<div class="alert-stable">No significant increases expected.</div>',
                        unsafe_allow_html=True)
    with al2:
        st.markdown("**Price Stable / Decreasing**")
        for comm, chg in stable[:6]:
            st.markdown(
                f'<div class="alert-stable"> <b>{comm}</b> '
                f'— Stable ({chg:+.1f}%) '
                f'(₱{predictions_next[comm]:.2f})</div>',
                unsafe_allow_html=True
            )
        for comm, chg in decreasing[:2]:
            st.markdown(
                f'<div class="alert-info"> Price drop ({chg:+.1f}%) '
                f'(₱{predictions_next[comm]:.2f})</div>',
                unsafe_allow_html=True
            )

    # Budget Planning Table
    st.markdown('<div class="section-header"> Budget Planning Guide — Next Month Forecast</div>',
                unsafe_allow_html=True)

    budget_df = pd.DataFrame({
        "Commodity": list(predictions_next.keys()),
        "Category":  [commodity_to_cat.get(c, "N/A") for c in predictions_next],
        "Current Forecast (₱)": list(predictions_now.values()),
        "Next Month Forecast (₱)": list(predictions_next.values()),
        "Change (%)": list(changes.values()),
    })
    budget_df["Recommendation"] = budget_df["Change (%)"].apply(
        lambda x: "Buy now if possible" if x > 5
                  else ("Safe to wait" if x < -2 else "No urgency")
    )
    budget_df = budget_df.sort_values("Change (%)", ascending=False).round(2)

    st.dataframe(
        budget_df.style.format({
            "Current Forecast (₱)":   "₱{:.2f}",
            "Next Month Forecast (₱)": "₱{:.2f}",
            "Change (%)":              "{:.1f}%",
        }).apply(
            lambda col: ["background: #FFF3E0" if "Buy now if possible" in str(v)
                         else ("background: #E8F5E9" if "Safe to wait" in str(v) else "")
                         for v in col],
            subset=["Recommendation"]
        ),
        use_container_width=True, height=400
    )

    # General insights
    st.markdown('<div class="section-header"> General Consumer Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-info">
    <b>For Households in Cavite:</b><br>
    • Stock up on staples (rice, oil, dried goods) when prices are forecasted to rise.<br>
    • Plan weekly menus around commodities showing stable or decreasing prices.<br>
    • Use the Forecasting page to check specific items before your shopping trip.
    </div>
    <br>
    <div class="alert-stable">
    <b>Seasonal Patterns to Know:</b><br>
    • Fish prices often rise during bad weather months (July–September typhoon season).<br>
    • Vegetables can spike during El Niño dry spells (March–May).<br>
    • Chicken and pork prices typically increase around the holidays (November–January).
    </div>
    <br>
    <div class="alert-increase">
    <b>Price Volatility Reminder:</b><br>
    • The ML models are trained on 2020–2026 historical data. External shocks (typhoons,
      fuel hikes, policy changes) are NOT automatically captured.<br>
    • Use predictions as a guide, not as an absolute guarantee.
    </div>
    """, unsafe_allow_html=True)


# SECTION 7 — ABOUT THE PROJECT


elif "About" in page:
    st.markdown("##  About the Project")

    st.markdown("""
    <div class="kpi-card">
    <h3 style="color:#4A3728; margin-top:0;">
        Analysis and Prediction of Monthly Food Prices in Cavite
        Using Machine Learning Techniques
    </h3>
    <p style="color:#5A4A3A;">
        A web-based food price forecasting and analytics system that analyzes
        historical monthly food prices in Cavite from 2020 to 2026 and predicts
        future food prices using machine learning regression models.
    </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###  Research Objectives")
        st.markdown("""
        1. Analyze historical food price data in Cavite (2020–2026)
        2. Visualize monthly and yearly food price trends
        3. Forecast future food prices using ML regression
        4. Compare performance of multiple regression algorithms
        5. Generate decision-support insights for consumers
        6. Present information via an interactive web dashboard
        """)

        st.markdown("###  Dataset Information")
        st.markdown(f"""
        - **Source** : WFP / HDX — Philippines Food Prices
        - **Provider** : World Food Programme (UN)
        - **Coverage** : Cavite, 2020–2026
        - **Records (filtered)** : {len(df):,} entries
        - **Commodities** : {df['commodity'].nunique()} tracked items
        - **Update Frequency** : Monthly
        - **License** : CC BY-IGO
        - **Download** : Kaggle (usmanlovescode)
        """)

    with c2:
        st.markdown("###  Technologies Used")
        st.markdown("""
        - **Frontend** : Streamlit, Plotly, Custom CSS
        - **Backend** : Python, Pandas, NumPy, Scikit-Learn
        - **Database** : SQLite
        - **ML Models** : Linear Regression, Decision Tree, Random Forest
        - **Dev Environment** : Google Colab / VS Code
        """)

        st.markdown("###  Scope and Limitations")
        st.markdown("""
        **In Scope:**
        - Food prices in Cavite province only
        - Monthly data from 2020 to 2026
        - Regression prediction (continuous price value)
        - Linear Regression, Decision Tree, Random Forest

        **Limitations:**
        - Geographic: Cavite only, not other Philippine regions
        - No real-time external factors (weather, oil prices, policy)
        - Accuracy depends on WFP dataset quality
        - Selected commodities only (55 items from WFP data)
        """)

    st.markdown("### Researchers")
    st.markdown("""
    <div class="kpi-card">
    <b>CPEN Group 2</b><br><br>
    Alvero, John Andrew B.<br>
    Brioso, Eriz Melvin V.<br>
    Costa, Joshua E.<br>
    Daygam, Daryl M.<br>
    Laxamana, Lorez Jhyner G.<br><br>
    <em>
    This system was developed as part of a capstone project to
    demonstrate the practical application of predictive analytics to
    real-world economic challenges in the Philippines.
    </em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###  References")
    st.markdown("""
    - PSA Philippines — Consumer Price Index (2026)
    - Trading Economics — Philippines Food Inflation (2026)
    - Pedregosa et al. — Scikit-learn: Machine Learning in Python, JMLR 12 (2011)
    - McKinney — Data Structures for Statistical Computing in Python (2010)
    - Géron, A. — Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2019)
    """)
