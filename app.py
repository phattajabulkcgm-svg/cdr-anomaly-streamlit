# =========================================
# app.py - Streamlit CDR Anomaly Dashboard
# =========================================

import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
import pytz

st.set_page_config(
    page_title="CDR Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CDR Anomaly Detection Dashboard")
st.markdown("""
ตรวจสอบความผิดปกติของ CDR โดยใช้ Prophet model + Rule-based
""")

# =========================================
# 1️⃣ Upload Excel
# =========================================
uploaded_file = st.file_uploader(
    "Upload Excel file (XLSX)", type=["xlsx"], accept_multiple_files=False
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True, errors='coerce')

    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # =========================================
    # 2️⃣ Predict & Train Date Inputs
    # =========================================
    col1, col2 = st.columns(2)

    with col1:
        default_predict_start = df['start_date'].max()
        predict_start_date = st.date_input(
            "Predict Start Date",
            value=default_predict_start
        )

    with col2:
        default_predict_end = df['end_date'].max()
        predict_end_date = st.date_input(
            "Predict End Date",
            value=default_predict_end
        )

    # แปลงให้เป็น datetime64[ns] สำหรับการกรอง
    predict_start_date = pd.to_datetime(predict_start_date)
    predict_end_date = pd.to_datetime(predict_end_date)

    # กำหนด train 6 เดือนย้อนหลัง
    train_start_date = predict_start_date - relativedelta(months=7)
    train_end_date = predict_end_date - relativedelta(months=2)

    st.markdown(f"**Train Range:** {train_start_date.date()} ถึง {train_end_date.date()}")

    # =========================================
    # 3️⃣ Event Filter Input
    # =========================================
    user_input = st.text_input(
        "data_masking:costcode list (comma-separated, costcode optional)",
        value=""
    )

    if user_input:
        event_pairs = []
        for pair in user_input.split(','):
            if ':' in pair:
                es, cc = pair.split(':', 1)
                event_pairs.append((es.strip(), cc.strip()))
            else:
                event_pairs.append((pair.strip(), None))

        # =========================================
        # 4️⃣ Prepare anomaly_results DataFrame
        # =========================================
        anomaly_results = pd.DataFrame(columns=[
            'predict_range','data_masking','account_num','event_type_id','costcode',
            'predicted_min','actual_volume','predicted_max',
            'results','diff','remark','train_range','method'
        ])

        # =========================================
        # 5️⃣ Loop over events
        # =========================================
        progress_bar = st.progress(0)
        total_events = len(event_pairs)

        for idx, (es, cc) in enumerate(event_pairs, start=1):
            df_es = df[df['data_masking'] == es]
            if cc:
                df_es = df_es[df_es['costcode'] == cc]

            if df_es.empty:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [None],
                    'event_type_id': [None],
                    'costcode': [cc],
                    'predicted_min': [0],
                    'actual_volume': [0],
                    'predicted_max': [0],
                    'results': [False],
                    'diff': [None],
                    'remark': ["❗ ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["No Data"]
                })], ignore_index=True)
                continue

            account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns and not df_es['account_num'].dropna().empty else None
            event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns and not df_es['event_type_id'].dropna().empty else None
            costcode = cc if cc else (df_es['costcode'].dropna().unique()[0] if not df_es.empty else None)

            # -----------------------------------------
            # actual / previous volume
            # -----------------------------------------
            actual_volume = df_es[
                (df_es['start_date'] >= predict_start_date) &
                (df_es['start_date'] <= predict_end_date)
            ]['volume_monthly'].sum()

            prev_volume = df_es[
                (df_es['start_date'] >= predict_start_date - relativedelta(months=1)) &
                (df_es['start_date'] <= predict_end_date - relativedelta(months=1))
            ]['volume_monthly'].sum()
            prev_volume = prev_volume if prev_volume != 0 else 0

            # -----------------------------------------
            # RULE 1: Usage ใหม่ (เดือนก่อน 0 เดือนนี้ >0)
            # -----------------------------------------
            if prev_volume == 0 and actual_volume > 0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [account_num],
                    'event_type_id': [event_type_id],
                    'costcode': [costcode],
                    'predicted_min': ['-'],
                    'actual_volume': [actual_volume],
                    'predicted_max': ['-'],
                    'results': [False],
                    'diff': ["100%"],
                    'remark': ["❗ เพิ่มขึ้น 100% จากเดือนก่อนหน้า"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["Rule-based"]
                })], ignore_index=True)
                continue

            # -----------------------------------------
            # RULE 2: Usage หายไป
            # -----------------------------------------
            if prev_volume > 0 and actual_volume == 0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                    'data_masking': [es],
                    'account_num': [account_num],
                    'event_type_id': [event_type_id],
                    'costcode': [costcode],
                    'predicted_min': [prev_volume],
                    'actual_volume': [0],
                    'predicted_max': [prev_volume],
                    'results': [False],
                    'diff': ["-100%"],
                    'remark': ["❗ Usage 0 ใน 2 เดือนล่าสุด"],
                    'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method': ["Rule-based"]
                })], ignore_index=True)
                continue

            # -----------------------------------------
            # Train Prophet model
            # -----------------------------------------
            df_train = df_es[
                (df_es['start_date'] >= train_start_date) &
                (df_es['start_date'] <= train_end_date)
            ].groupby('start_date')['volume_monthly'].sum().reset_index()
            df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

            if df_train.shape[0] >= 6:
                model = Prophet()
                model.fit(df_train)
                forecast = model.predict(pd.DataFrame({'ds':[predict_start_date]}))
                predicted_min = max(df_train['y'].min(), forecast['yhat_lower'].values[0])
                predicted_max = min(df_train['y'].max(), forecast['yhat_upper'].values[0])
                method = "Prophet"
            else:
                predicted_min = df_train['y'].min() if not df_train.empty else 0
                predicted_max = df_train['y'].max() if not df_train.empty else 0
                method = "Median fallback"

            # -----------------------------------------
            # Determine results and diff
            # -----------------------------------------
            if actual_volume == 0:
                results = False
                remark = "❗ Usage 0 ใน 2 เดือนล่าสุด"
            elif actual_volume < 200:
                results = True
                remark = "ปกติ (CDR < 200)"
            elif predicted_min <= actual_volume <= predicted_max:
                results = True
                remark = "diff แต่ยังอยู่ใน threshold"
            else:
                # คำนวณ % diff ตาม threshold
                diff_val = round((actual_volume - predicted_max)/predicted_max*100,2) if actual_volume > predicted_max else round((actual_volume - predicted_min)/predicted_min*100,2)
                if actual_volume <= 1000 and abs(diff_val) >= 80:
                    results = False
                elif actual_volume <= 100_000 and abs(diff_val) >= 50:
                    results = False
                elif actual_volume <= 1_000_000 and abs(diff_val) >= 20:
                    results = False
                else:
                    results = True
                remark = "diff แต่เกิน threshold" if not results else "diff แต่ยังอยู่ใน threshold"

            diff = round((actual_volume - predicted_max)/predicted_max*100,2) if predicted_max > 0 else None

            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'predict_range': [f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
                'data_masking': [es],
                'account_num': [account_num],
                'event_type_id': [event_type_id],
                'costcode': [costcode],
                'predicted_min': [predicted_min],
                'actual_volume': [actual_volume],
                'predicted_max': [predicted_max],
                'results': [results],
                'diff': [f"{diff}%" if diff is not None else None],
                'remark': [remark],
                'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                'method': [method]
            })], ignore_index=True)

            progress_bar.progress(idx/total_events)

        # =========================================
        # Sort results by results & remark
        # =========================================
        def remark_priority(x):
            if "❗" in str(x): return 0
            elif "diff แต่เกิน" in str(x): return 1
            elif "diff แต่ยัง" in str(x): return 2
            else: return 3

        anomaly_results['results_sort'] = anomaly_results['results'].apply(lambda x: 1 if x else 0)
        anomaly_results['remark_sort'] = anomaly_results['remark'].apply(remark_priority)
        anomaly_results = anomaly_results.sort_values(by=['results_sort','remark_sort'], ascending=[True,True])
        anomaly_results = anomaly_results.drop(columns=['results_sort','remark_sort'])

        # =========================================
        # Display & Download
        # =========================================
        st.markdown("### Results")
        st.dataframe(anomaly_results)

        # Download
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        anomaly_results.to_excel(file_name, index=False)
        st.download_button(
            label="📥 Download Excel",
            data=open(file_name, "rb").read(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
