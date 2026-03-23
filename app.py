# =========================================
# app.py - Streamlit Complete Version (data_masking only)
# =========================================
import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="CDR SMS Bulk Detection Dashboard", layout="wide")
st.title("📊 CDR SMS Bulk Detection Dashboard")

# ==========================
# STEP 1️⃣ Upload File & Select Predict Period
# ==========================
st.header("STEP 1: Upload File & Select Predict Period")
st.info("""
**Required columns in Excel:**  
`data_masking`, `start_date`, `volume_monthly`, `account_num`, `event_type_id`  
Make sure column names match exactly (case-insensitive)
""")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
col1, col2 = st.columns(2)
with col1:
    predict_start_date = st.date_input("Predict Start Date", datetime.today())
with col2:
    predict_end_date = st.date_input("Predict End Date", datetime.today())

train_start_date = predict_start_date - relativedelta(months=7)
train_end_date   = predict_end_date   - relativedelta(months=2)

# ==========================
# STEP 2️⃣ Enter data_masking & Run Anomaly Detection
# ==========================
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')

    st.success("✅ File loaded successfully!")

    st.header("STEP 2: Enter data_masking list (comma-separated)")
    data_masking_input = st.text_area(
        "Enter data_masking (e.g., A1,A11,A98)", 
        ",".join(df['data_masking'].dropna().unique())
    )

    if st.button("▶ Run Anomaly Detection"):
        event_list = [x.strip() for x in data_masking_input.split(',') if x.strip()]
        anomaly_results = pd.DataFrame(columns=[
            'predict_range','data_masking','account_num','event_type_id',
            'predicted_min','actual_volume','predicted_max',
            'results','diff','remark','train_range','method'
        ])

        with st.spinner("⏳ Calculating anomalies..."):
            for es in event_list:
                df_es = df[df['data_masking'] == es]

                if df_es.empty:
                    anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                        'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                        'data_masking':[es],'account_num':[None],'event_type_id':[None],
                        'predicted_min':[0],'actual_volume':[0],'predicted_max':[0],
                        'results':[False],'diff':[None],'remark':["ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"],
                        'train_range':[f"{train_start_date} ถึง {train_end_date}"],'method':["No Data"]
                    })], ignore_index=True)
                    continue

                account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns and not df_es['account_num'].dropna().empty else None
                event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns and not df_es['event_type_id'].dropna().empty else None

                actual_volume = df_es[
                    (df_es['start_date'] >= pd.to_datetime(predict_start_date)) &
                    (df_es['start_date'] <= pd.to_datetime(predict_end_date))
                ]['volume_monthly'].sum()

                prev_volume = df_es[
                    (df_es['start_date'] >= pd.to_datetime(predict_start_date) - relativedelta(months=1)) &
                    (df_es['start_date'] <= pd.to_datetime(predict_end_date) - relativedelta(months=1))
                ]['volume_monthly'].sum()

                df_train = df_es[
                    (df_es['start_date'] >= pd.to_datetime(train_start_date)) &
                    (df_es['start_date'] <= pd.to_datetime(train_end_date))
                ].groupby('start_date')['volume_monthly'].sum().reset_index()
                df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

                if df_train.shape[0] >= 6:
                    model = Prophet()
                    model.fit(df_train)
                    forecast = model.predict(pd.DataFrame({'ds':[pd.to_datetime(predict_start_date)]}))
                    predicted_min = max(df_train['y'].min(), forecast['yhat_lower'].values[0])
                    predicted_max = min(df_train['y'].max(), forecast['yhat_upper'].values[0])
                    method_val = "Prophet"
                else:
                    predicted_min = df_train['y'].min() if not df_train.empty else 0
                    predicted_max = df_train['y'].max() if not df_train.empty else 0
                    method_val = "Median fallback"

                # rules & results
                if prev_volume==0 and actual_volume>0:
                    results_val = False
                    diff_val = 100
                    remark_txt = "❗ มี Usage ใหม่ (เดือนก่อน = 0)"
                elif actual_volume==0:
                    if prev_volume==0:
                        remark_txt="❗ Usage 0 ใน 2 เดือนล่าสุด"
                    else:
                        remark_txt="❗ Usage หายไปจากเดือนก่อน"
                    results_val = False
                    diff_val = -100
                elif actual_volume < 200:
                    results_val = True
                    diff_val = 0
                    remark_txt = "ปกติ (CDR < 200)"
                elif predicted_min <= actual_volume <= predicted_max:
                    results_val = True
                    diff_val = 0
                    remark_txt = "Diff แต่ยังอยู่ใน threshold"
                else:
                    if actual_volume > predicted_max:
                        diff_val = round((actual_volume - predicted_max)/predicted_max*100,2)
                    else:
                        diff_val = round((actual_volume - predicted_min)/predicted_min*100,2) if predicted_min>0 else 0
                    abs_val = abs(diff_val)
                    if 1 <= actual_volume <= 1_000:
                        results_val = False if abs_val>=80 else True
                    elif 1_001 <= actual_volume <= 100_000:
                        results_val = False if abs_val>=50 else True
                    elif 100_001 <= actual_volume <= 1_000_000:
                        results_val = False if abs_val>=20 else True
                    elif actual_volume >= 1_000_001:
                        results_val = False
                    remark_txt = "❗ เพิ่มขึ้นผิดปกติ" if actual_volume>predicted_max else "❗ ลดลงผิดปกติ"

                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                    'data_masking':[es],
                    'account_num':[account_num],
                    'event_type_id':[event_type_id],
                    'predicted_min':[predicted_min],
                    'actual_volume':[actual_volume],
                    'predicted_max':[predicted_max],
                    'results':[results_val],
                    'diff':[f"{diff_val}%" if diff_val is not None else None],
                    'remark':[remark_txt],
                    'train_range':[f"{train_start_date} ถึง {train_end_date}"],
                    'method':[method_val]
                })], ignore_index=True)

        # sort & TRUE/FALSE
        anomaly_results['results_sort'] = anomaly_results['results'].apply(lambda x: 1 if x else 0)
        anomaly_results = anomaly_results.sort_values(by=['results_sort','remark'], ascending=[True,True]).drop(columns=['results_sort'])
        anomaly_results['results'] = anomaly_results['results'].apply(lambda x: 'TRUE' if x else 'FALSE')

        # highlight
        def highlight_results(val):
            color = 'lightgreen' if val == 'TRUE' else 'lightcoral'
            return f'background-color: {color}'

        st.subheader("STEP 3: Anomaly Results Table")
        st.dataframe(anomaly_results.style.applymap(highlight_results, subset=['results']))

        # download
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        anomaly_results.to_excel(file_name,index=False)
        st.download_button("💾 Download Excel", file_name, file_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==========================
# STEP 4️⃣ Select Event to View Trend Graph
# ==========================
if uploaded_file is not None:
    st.header("STEP 4: Select Event to View Trend Graph")
    data_masking_list = df['data_masking'].dropna().unique().tolist()
    selected_event = st.selectbox("Select data_masking to view trend", data_masking_list)
    df_event = df[df['data_masking'] == selected_event].copy()

    if not df_event.empty:
        st.subheader(f"📈 Trend for {selected_event}")
        df_trend = df_event.groupby('start_date')['volume_monthly'].sum().reset_index()
        df_trend.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

        if df_trend.shape[0] >= 2:
            model = Prophet()
            model.fit(df_trend)
            future = model.make_future_dataframe(periods=1, freq='M')
            forecast = model.predict(future)

            st.line_chart(pd.DataFrame({
                'Actual': df_trend.set_index('ds')['y'],
                'Forecast': forecast.set_index('ds')['yhat'],
                'Lower': forecast.set_index('ds')['yhat_lower'],
                'Upper': forecast.set_index('ds')['yhat_upper']
            }))
        else:
            st.warning("ข้อมูลไม่เพียงพอสำหรับการสร้าง trend (ต้องมีอย่างน้อย 2 เดือน)")
    else:
        st.warning("ไม่มีข้อมูลสำหรับ event ที่เลือก")
