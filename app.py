# =========================================
# 📊 CDR Bulk Top10 Anomaly Detection | CGV (Dark Mode Toggle + Default Dates)
# =========================================
import streamlit as st
import pandas as pd
from io import BytesIO
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pytz

st.set_page_config(page_title="CDR Bulk Top10 Anomaly Detection | CGV", layout="wide")

# ==============================
# Dark Mode toggle
# ==============================
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0d1117; color: #e6ffe6; }
        .stButton>button { background-color: #4CAF50; color: white; font-weight:bold; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
        .stFileUploader>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
        .css-1d391kg { color: #e6ffe6; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stButton>button { background-color: #4CAF50; color: white; font-weight:bold; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input { background-color: #e6ffe6; color: black; }
        .stFileUploader>div>div>input { background-color: #e6ffe6; color: black; }
        </style>
    """, unsafe_allow_html=True)

# ==============================
# Title
# ==============================
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 CDR Bulk Top10 Anomaly Detection | CGV</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# STEP 1: Upload Excel
# ==============================
st.markdown("### Step 1️⃣ Upload Excel File")
uploaded_file = st.file_uploader("📁 Upload Excel file (XLSX)", type=["xlsx"], label_visibility="visible")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')

    # ==============================
    # STEP 2: Predict Range & Data Masking (Default min/max latest month)
    # ==============================
    st.markdown("### Step 2️⃣ Set Predict Range & Data Masking")
    latest_month = df['start_date'].max()
    month_start = latest_month.replace(day=1)
    month_end = latest_month

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            predict_start_date = st.date_input("📅 Predict Start Date", value=month_start)
        with col2:
            predict_end_date = st.date_input("📅 Predict End Date", value=month_end)

        data_masking_input = st.text_area(
            "💠 Data Masking (comma-separated, e.g. A1, A100, ...)", height=150
        )

    # ==============================
    # STEP 3: Run anomaly detection
    # ==============================
    run_button = st.button("🚀 Step 3️⃣ Run Anomaly Detection")

    if run_button and data_masking_input:
        st.info("Processing... This may take a few seconds ⏳")
        # ... logic เดิม Prophet + Rule-based ...
