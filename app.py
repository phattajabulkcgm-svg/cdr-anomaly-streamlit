# =========================================
# 📊 CDR Bulk Top10 Anomaly Detection | CGV (Black Theme + Default Dates)
# =========================================
import streamlit as st
import pandas as pd
from io import BytesIO
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pytz

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="CDR Bulk Top10 Anomaly Detection | CGV",
    layout="wide"
)

# ==============================
# Black Theme CSS
# ==============================
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #e6ffe6; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight:bold; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stDateInput>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
    .stFileUploader>div>div>input { background-color: #1c1c1c; color: #e6ffe6; border: 1px solid #4CAF50; }
    .css-1d391kg { color: #e6ffe6; }
    .stDataFrame div.row_heading.level0 { color: #4CAF50; font-weight:bold; }
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
        data_masking_selected = [x.strip() for x in data_masking_input.split(",") if x.strip()]
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

        for i, es in enumerate(data_masking_selected, 1):
            df_es = df[df['data_masking'] == es]

            if df_es.empty:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [None],
                    'event_type_id': [None],
                    'costcode': [None],
                    'predicted_min': [None],
                    'actual_volume': [0],
                    'predicted_max': [None],
                    'is_nomaly': [False],
                    'diff': [None],
                    'level': [None],
                    'remark': ["⚫ ไม่มีข้อมูลย้อนหลัง 6 เดือน"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["No Data"]
                })], ignore_index=True)
                progress.progress(i/total)
                continue

            # ... (Prophet + Rule-based logic เดิม) ...

        # ==============================
        # STEP 4: Show Results + Download
        # ==============================
        st.markdown("### Step 4️⃣ ✅ Anomaly Results")
        st.dataframe(anomaly_results)
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        output = BytesIO()
        file_name = f"cdr_bulk_top10_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        anomaly_results.to_excel(output, index=False)
        output.seek(0)
        st.download_button("📥 Download Result Excel", data=output, file_name=file_name)
