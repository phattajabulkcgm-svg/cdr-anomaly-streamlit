# =========================================
# 📊 CDR SMS Bulk Top10 Dashboard | CGV
# Dark/Light Mode + SMS Theme
# Predict Start Date = latest, Predict End Date = max
# =========================================

import streamlit as st
import pandas as pd
from io import BytesIO
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pytz
import plotly.express as px

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="CDR Bulk Top10 Dashboard | CGV",
    layout="wide"
)

# ==============================
# Dark/Light Mode toggle
# ==============================
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #0d1117; color: #e6ffe6; }
        .stButton>button { background-color: #4CAF50; color: white; font-weight:bold; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
        .stFileUploader>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: black; }
        .stButton>button { background-color: #4CAF50; color: white; font-weight:bold; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input { background-color: #e6ffe6; color: black; border: 1px solid #4CAF50; }
        .stFileUploader>div>div>input { background-color: #e6ffe6; color: black; border: 1px solid #4CAF50; }
        </style>
    """, unsafe_allow_html=True)

# ==============================
# Title
# ==============================
st.markdown("""
    <h1 style='text-align:center; color:#4CAF50;'>📊 CDR Bulk Top10 Dashboard | CGV</h1>
    <p style='text-align:center; color:gray;'>Monitor SMS / CDR anomalies in bulk Top 10 </p>
""", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# Step 1️⃣ Upload Excel
# ==============================
st.markdown("### Step 1️⃣ Upload Excel File")
uploaded_file = st.file_uploader("📁 Upload Excel file (XLSX)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True, errors='coerce')

    # ==============================
    # Step 2️⃣ Predict Range & Data Masking
    # ==============================
    st.markdown("### Step 2️⃣ Set Predict Range & Data Masking")

    # Predict Start = latest start_date, Predict End = max end_date
    predict_start_date = df['start_date'].max().date()
    predict_end_date = df['end_date'].max().date() if 'end_date' in df.columns else df['start_date'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"📅 Predict Start Date: **{predict_start_date}**")
    with col2:
        st.markdown(f"📅 Predict End Date: **{predict_end_date}**")

    # Multi-select Data Masking
    data_masking_options = df['data_masking'].dropna().unique().tolist()
    data_masking_selected = st.multiselect("💠 Select Data Masking", options=data_masking_options, default=data_masking_options[:10])

    # ==============================
    # Step 3️⃣ Run anomaly detection
    # ==============================
    run_button = st.button("🚀 Step 3️⃣ Run Anomaly Detection")

    if run_button and data_masking_selected:
        st.info("Processing... ⏳")
        train_start_date = pd.to_datetime(predict_start_date) - relativedelta(months=7)
        train_end_date   = pd.to_datetime(predict_end_date) - relativedelta(months=2)
        predict_start_date = pd.to_datetime(predict_start_date)
        predict_end_date   = pd.to_datetime(predict_end_date)

        anomaly_results = pd.DataFrame(columns=[
            'predict_range','data_masking','account_num','event_type_id','costcode',
            'predicted_min','actual_volume','predicted_max',
            'is_nomaly','diff','level','remark',
            'train_range','method'
        ])

        progress = st.progress(0)
        total = len(data_masking_selected)

        for i, es in enumerate(data_masking_selected,1):
            df_es = df[df['data_masking']==es]
            # (Logic เดิมทั้งหมดเหมือนตัวก่อนหน้า)
            # ... (คุณสามารถ copy loop จากโค้ดตัวก่อนหน้า)
