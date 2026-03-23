# app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

st.title("CDR Anomaly Detection")

# =========================
# 1️⃣ Upload file
# =========================
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True, errors='coerce')

    # =========================
    # 2️⃣ Set Predict Range (default จากไฟล์) + allow edit
    # =========================
    default_predict_start = df['start_date'].max()
    default_predict_end = df['end_date'].max()

    predict_start_date = st.date_input(
        "Predict Start Date",
        value=default_predict_start,
        min_value=df['start_date'].min(),
        max_value=df['end_date'].max()
    )
    predict_end_date = st.date_input(
        "Predict End Date",
        value=default_predict_end,
        min_value=df['start_date'].min(),
        max_value=df['end_date'].max()
    )

    # =========================
    # 3️⃣ Train Range (6 เดือนย้อนหลัง + skip 1)
    # =========================
    train_start_date = predict_start_date - relativedelta(months=7)
    train_end_date = predict_end_date - relativedelta(months=2)

    st.write(f"Train Start Date: {train_start_date}")
    st.write(f"Train End Date: {train_end_date}")

    # =========================
    # 4️⃣ Input costcode / event
    # =========================
    user_input = st.text_input("data_masking:costcode list (comma-separated, costcode optional)", "")
    event_pairs = []
    for pair in user_input.split(','):
        if ':' in pair:
            es, cc = pair.split(':', 1)
            event_pairs.append((es.strip(), cc.strip()))
        elif pair.strip():
            event_pairs.append((pair.strip(), None))

    # =========================
    # 5️⃣ Prepare result table
    # =========================
    anomaly_results = pd.DataFrame(columns=[
        'data_masking','account_num','event_type_id','costcode',
        'predicted_min','actual_volume','predicted_max','results','diff','remark','train_range','method'
    ])

    for es, cc in event_pairs:
        df_es = df[df['data_masking'] == es]
        if cc:
            df_es = df_es[df_es['costcode'] == cc]

        if df_es.empty:
            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'data_masking':[es],
                'account_num':[None],
                'event_type_id':[None],
                'costcode':[cc],
                'predicted_min':[0],
                'actual_volume':[0],
                'predicted_max':[0],
                'results':[False],
                'diff':[None],
                'remark':["ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"],
                'train_range':[f"{train_start_date} ถึง {train_end_date}"],
                'method':["No Data"]
            })], ignore_index=True)
            continue

        account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns and not df_es['account_num'].dropna().empty else None
        event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns and not df_es['event_type_id'].dropna().empty else None
        costcode_val = cc if cc else (df_es['costcode'].dropna().unique()[0] if not df_es.empty else None)

        # actual / prev
        actual_volume = df_es[(df_es['start_date'] >= predict_start_date) & (df_es['start_date'] <= predict_end_date)]['volume_monthly'].sum()
        prev_volume = df_es[(df_es['start_date'] >= predict_start_date - relativedelta(months=1)) & (df_es['start_date'] <= predict_end_date - relativedelta(months=1))]['volume_monthly'].sum()
        prev_volume = prev_volume if prev_volume != 0 else None

        # =========================
        # Train model
        # =========================
        df_train = df_es[(df_es['start_date'] >= train_start_date) & (df_es['start_date'] <= train_end_date)].groupby('start_date')['volume_monthly'].sum().reset_index()
        df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

        if not df_train.empty and df_train.shape[0]>=6:
            model = Prophet()
            model.fit(df_train)
            forecast = model.predict(pd.DataFrame({'ds':[predict_start_date]}))
            predicted_min = max(0, min(forecast['yhat_lower'].values[0], df_train['y'].min()))
            predicted_max = max(forecast['yhat_upper'].values[0], df_train['y'].max())
            method="Prophet"
        else:
            median_val = df_train['y'].median() if not df_train.empty else 0
            predicted_min = df_train['y'].min() if not df_train.empty else 0
            predicted_max = median_val
            method="Median fallback"

        # =========================
        # Anomaly logic
        # =========================
        diff_val = None
        remark_text = ""
        result_flag = True

        # actual_volume = 0
        if actual_volume == 0:
            if prev_volume in [0,None]:
                remark_text="Usage 0 ใน 2 เดือนล่าสุด"
            else:
                remark_text="❗ Usage หายไปจากเดือนก่อน"
            result_flag=False
            predicted_min = df_es['volume_monthly'].max() if not df_es.empty else 0
            predicted_max = predicted_min

        # actual_volume < 200
        elif actual_volume < 200:
            remark_text="ปกติ (CDR น้อยกว่า 200)"
            result_flag=True

        # actual_volume in min-max
        elif predicted_min <= actual_volume <= predicted_max:
            remark_text="Diff แต่ยังอยู่ใน threshold"
            result_flag=True

        # actual_volume out of min-max
        else:
            if actual_volume <= 1000:
                threshold=0.8
            elif actual_volume <= 100_000:
                threshold=0.5
            elif actual_volume <= 1_000_000:
                threshold=0.2
            else:
                threshold=0  # >1M → ทุก % ถือ False

            diff_val = round((actual_volume - predicted_max)/predicted_max*100,2) if predicted_max>0 else 0
            if abs(diff_val)/100 > threshold:
                result_flag=False
                remark_text="❗ เพิ่มขึ้นผิดปกติ" if actual_volume > predicted_max else "❗ ลดลงผิดปกติ"
            else:
                remark_text="Diff แต่ยังอยู่ใน threshold"
                result_flag=True

        anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
            'data_masking':[es],
            'account_num':[account_num],
            'event_type_id':[event_type_id],
            'costcode':[costcode_val],
            'predicted_min':[predicted_min],
            'actual_volume':[actual_volume],
            'predicted_max':[predicted_max],
            'results':[result_flag],
            'diff':[f"{diff_val}%" if diff_val is not None else None],
            'remark':[remark_text],
            'train_range':[f"{train_start_date} ถึง {train_end_date}"],
            'method':[method]
        })], ignore_index=True)

    # =========================
    # Sort by FALSE → remark
    # =========================
    anomaly_results['remark_sort'] = anomaly_results['remark'].apply(lambda x: 0 if "❗" in str(x) else 1)
    anomaly_results = anomaly_results.sort_values(by=['results','remark_sort'], ascending=[True, True]).drop(columns=['remark_sort'])

    st.write(anomaly_results)

    # =========================
    # Download
    # =========================
    tz = pytz.timezone('Asia/Bangkok')
    now = datetime.now(tz)
    file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
    anomaly_results.to_excel(file_name, index=False)
    st.download_button("Download Excel", file_name, file_name)
