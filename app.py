# =========================================
# 📊 CDR Bulk Dashboard | CGV
# Prophet + Rule-based | Dark/Light Mode | Full Table
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
    page_title="CDR Bulk Dashboard | CGV",
    layout="wide"
)

# ==============================
# Dark Mode toggle (default Light)
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
    <h1 style='text-align:center; color:#4CAF50;'>📊 CDR Bulk Top 10 Dashboard | CGV</h1>
    <p style='text-align:center; color:gray;'>Monitor SMS / CDR anomalies in bulk</p>
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

    predict_start_default = df['start_date'].max().date()
    predict_end_default = df['end_date'].max().date() if 'end_date' in df.columns else df['start_date'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        predict_start_date = st.date_input("📅 Predict Start Date", value=predict_start_default)
    with col2:
        predict_end_date = st.date_input("📅 Predict End Date", value=predict_end_default)

    data_masking_input = st.text_area(
        "💠 Data Masking (comma-separated, e.g. A1:costcode, A2:costcode, ...)", height=150
    )

    run_button = st.button("🚀 Step 3️⃣ Run Anomaly Detection")

    if run_button and data_masking_input:
        st.info("Processing... ⏳")
        event_pairs = []
        for pair in data_masking_input.split(','):
            if ':' in pair:
                es, cc = pair.split(':', 1)
                event_pairs.append((es.strip(), cc.strip()))
            else:
                event_pairs.append((pair.strip(), None))

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

        for es, cc in event_pairs:
            df_es = df[df['data_masking'] == es]
            if cc:
                df_es = df_es[df_es['costcode'] == cc]

            # ❌ ไม่มีข้อมูล
            if df_es.empty:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [None],
                    'event_type_id': [None],
                    'costcode': [cc],
                    'predicted_min': [None],
                    'actual_volume': [0],
                    'predicted_max': [None],
                    'is_nomaly': [False],
                    'diff': [None],
                    'level': [None],
                    'remark': ["⚫ ไม่มีข้อมูลของเบอร์นี้ ย้อนหลัง 6 เดือน"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["No Data"]
                })], ignore_index=True)
                continue

            account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns else None
            event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns else None
            costcode = cc if cc else df_es['costcode'].dropna().unique()[0]

            # =========================
            # actual / prev
            # =========================
            actual_volume = df_es[
                (df_es['start_date'] >= predict_start_date) &
                (df_es['start_date'] <= predict_end_date)
            ]['volume_monthly'].sum()

            prev_volume = df_es[
                (df_es['start_date'] >= predict_start_date - relativedelta(months=1)) &
                (df_es['start_date'] <= predict_end_date - relativedelta(months=1))
            ]['volume_monthly'].sum()

            prev_volume = prev_volume if prev_volume != 0 else None

            # =========================================
            # RULE 1: New Usage
            # =========================================
            if (prev_volume is None or prev_volume == 0) and actual_volume > 0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [account_num],
                    'event_type_id': [event_type_id],
                    'costcode': [costcode],
                    'predicted_min': [None],
                    'actual_volume': [actual_volume],
                    'predicted_max': [None],
                    'is_nomaly': [False],
                    'diff': ["100%"],
                    'level': ["High"],
                    'remark': ["❗ มี Usage ใหม่ (เดือนก่อน = 0)"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["Rule-based"]
                })], ignore_index=True)
                continue

            # =========================================
            # train model
            # =========================================
            df_train = df_es[
                (df_es['start_date'] >= train_start_date) &
                (df_es['start_date'] <= train_end_date)
            ]
            df_train = df_train.groupby('start_date')['volume_monthly'].sum().reset_index()
            df_train.rename(columns={'start_date': 'ds', 'volume_monthly': 'y'}, inplace=True)

            if df_train.shape[0] >= 6:
                model = Prophet()
                model.fit(df_train)
                forecast = model.predict(pd.DataFrame({'ds': [predict_start_date]}))
                predicted_min = max(0, forecast['yhat_lower'].values[0])
                predicted_max = max(predicted_min, forecast['yhat_upper'].values[0])
                method = "Prophet"
            else:
                median_val = df_train['y'].median() if not df_train.empty else 1
                predicted_min = 0
                predicted_max = median_val
                method = "Median fallback"

            if predicted_max < 1:
                predicted_max = max(df_train['y'].median(), 1) if not df_train.empty else 1
                predicted_min = 0

            # =========================================
            # RULE 2: Drop to Zero
            # =========================================
            if prev_volume is not None and prev_volume > 0 and actual_volume == 0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [account_num],
                    'event_type_id': [event_type_id],
                    'costcode': [costcode],
                    'predicted_min': [predicted_min],
                    'actual_volume': [0],
                    'predicted_max': [predicted_max],
                    'is_nomaly': [False],
                    'diff': ["-100%"],
                    'level': ["Low"],
                    'remark': ["❗ Usage หายไปจากเดือนก่อน"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': [method]
                })], ignore_index=True)
                continue

            # =========================================
            # anomaly logic
            # =========================================
            diff = round((actual_volume - predicted_max) / predicted_max * 100, 2) if predicted_max else None
            if diff is None:
                is_nomaly = False
                level = None
                remark = ""
            elif diff == 0:
                is_nomaly = True
                level = "Normal"
                remark = ""
            else:
                if diff > 50:
                    level = "High"
                    remark = "❗ เพิ่มขึ้นผิดปกติ เทียบ 6 เดือนย้อนหลัง"
                elif diff < -50:
                    level = "Low"
                    remark = "❗ ลดลงผิดปกติ เทียบ 6 เดือนย้อนหลัง"
                else:
                    level = "Normal"
                    remark = ""
                is_nomaly = abs(diff) > 5

            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                'data_masking': [es],
                'account_num': [account_num],
                'event_type_id': [event_type_id],
                'costcode': [costcode],
                'predicted_min': [predicted_min],
                'actual_volume': [actual_volume],
                'predicted_max': [predicted_max],
                'is_nomaly': [is_nomaly],
                'diff': [f"{diff}%" if diff is not None else None],
                'level': [level],
                'remark': [remark],
                'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                'method': [method]
            })], ignore_index=True)

        # ==============================
        # sort table
        # ==============================
        def remark_priority(x):
            if "❗" in str(x):
                return 0
            elif "🟡" in str(x):
                return 1
            elif "⚫" in str(x):
                return 2
            else:
                return 3

        anomaly_results['is_nomaly_sort'] = anomaly_results['is_nomaly'].apply(lambda x: 0 if x == False else 1)
        anomaly_results['remark_sort'] = anomaly_results['remark'].apply(remark_priority)
        anomaly_results = anomaly_results.sort_values(by=['is_nomaly_sort','remark_sort']).drop(columns=['is_nomaly_sort','remark_sort'])

        # ==============================
        # show table
        # ==============================
        st.dataframe(anomaly_results)

        # ==============================
        # download
        # ==============================
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        output = BytesIO()
        file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        anomaly_results.to_excel(output, index=False)
        output.seek(0)
        st.download_button("📥 Download Excel", data=output, file_name=file_name)
