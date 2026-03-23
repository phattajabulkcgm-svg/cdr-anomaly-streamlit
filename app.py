import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

st.set_page_config(page_title="CDR Anomaly Detection", layout="wide")

st.title("📊 CDR Anomaly Detection")

# 1️⃣ Upload Excel
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    st.success(f"File uploaded: {uploaded_file.name}")

    # 2️⃣ Auto-fill predict range from max dates
    max_start = pd.to_datetime(df['start_date'], dayfirst=True).max()
    max_end   = pd.to_datetime(df['end_date'], dayfirst=True).max()

    predict_start_date = st.date_input("Predict Start Date", max_start)
    predict_end_date   = st.date_input("Predict End Date", max_end)

    # 3️⃣ Train range (6 เดือนย้อนหลัง)
    train_start_date = predict_start_date - relativedelta(months=7)
    train_end_date   = predict_end_date   - relativedelta(months=2)
    st.text(f"Train Start Date: {train_start_date}")
    st.text(f"Train End Date: {train_end_date}")

    # 4️⃣ Input costcode list
    costcode_input = st.text_input("data_masking:costcode list (comma-separated, costcode optional)")
    event_pairs = []
    for pair in costcode_input.split(','):
        if ':' in pair:
            es, cc = pair.split(':', 1)
            event_pairs.append((es.strip(), cc.strip()))
        else:
            event_pairs.append((pair.strip(), None))

    # 5️⃣ Process anomalies
    anomaly_results = pd.DataFrame(columns=[
        'predict_range','data_masking','account_num','event_type_id','costcode',
        'predicted_min','actual_volume','predicted_max',
        'results','diff','remark','train_range','method'
    ])

    for es, cc in event_pairs:
        df_es = df[df['data_masking'] == es]
        if cc:
            df_es = df_es[df_es['costcode'] == cc]

        if df_es.empty:
            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'predict_range': [f"{predict_start_date} ถึง {predict_end_date}"],
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

        # … ต่อโค้ด Prophet / median fallback / logic results TRUE/FALSE …
        # ⚠️ keep results column TRUE/FALSE, remark, predicted_min/max
        # ⚠️ ปรับให้ predicted_min ไม่เกิน min ของข้อมูลย้อนหลัง
        # ⚠️ diff = 0% ถ้า actual_volume อยู่ใน min-max
        # ⚠️ ถ้า actual_volume = 0 → results = FALSE, remark = "❗ Usage หายไปจากเดือนก่อน" หรือ "Usage 0 ใน 2 เดือนล่าสุด"
        # ⚠️ ถ้าเดือนนี้ไม่มีและเดือนก่อนก็ไม่มี → results = FALSE, remark = "ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"

    # 6️⃣ Show results
    st.subheader("Anomaly Results")
    st.dataframe(anomaly_results)

    # 7️⃣ Download option
    tz = pytz.timezone('Asia/Bangkok')
    now = datetime.now(tz)
    file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
    anomaly_results.to_excel(file_name, index=False)
    st.download_button("Download Report", data=open(file_name, "rb"), file_name=file_name)
