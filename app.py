# streamlit_app.py
# =========================================
# 0️⃣ install (เฉพาะ Colab/Cloud ใช้เฉพาะครั้งแรก)
# =========================================
# !pip install streamlit prophet==1.2.1 cmdstanpy==1.1.0 -q

# =========================================
# 1️⃣ import
# =========================================
import streamlit as st
import pandas as pd
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pytz

# =========================================
# 2️⃣ Streamlit UI
# =========================================
st.set_page_config(page_title="CDR Bulk Anomaly Dashboard", layout="wide")
st.title("📊 CDR Bulk Anomaly Dashboard | CGV")
st.write("Monitor SMS / CDR anomalies in bulk data")

# 2.1 Upload Excel
uploaded_file = st.file_uploader("📁 Upload Excel file (XLSX)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')
    
    # 2.2 Predict range
    col1, col2 = st.columns(2)
    with col1:
        predict_start_date = st.date_input("📅 Predict Start Date", df['start_date'].max())
    with col2:
        predict_end_date = st.date_input("📅 Predict End Date", df['start_date'].max())
    
    # 2.3 Data masking input
    data_masking_input = st.text_input("💠 Data Masking (comma-separated, e.g. A1,A100,A101)", value="A1")
    
    if st.button("🚀 Run Anomaly Detection"):
        # =========================================
        # 3️⃣ Prepare event pairs
        # =========================================
        event_pairs = []
        for pair in data_masking_input.split(','):
            if ':' in pair:
                es, cc = pair.split(':',1)
                event_pairs.append((es.strip(), cc.strip()))
            else:
                event_pairs.append((pair.strip(), None))
        
        # =========================================
        # 4️⃣ Set training range
        # =========================================
        train_start_date = pd.to_datetime(predict_start_date) - relativedelta(months=7)
        train_end_date   = pd.to_datetime(predict_end_date) - relativedelta(months=2)
        
        # =========================================
        # 5️⃣ Prepare results table
        # =========================================
        anomaly_results = pd.DataFrame(columns=[
            'predict_range','data_masking','account_num','event_type_id','costcode',
            'predicted_min','actual_volume','predicted_max',
            'is_nomaly','status','remark',
            'train_range','method'
        ])
        
        # =========================================
        # 6️⃣ Loop each event
        # =========================================
        for es, cc in event_pairs:
            df_es = df[df['data_masking'] == es]
            if cc:
                df_es = df_es[df_es['costcode'] == cc]

            # ❌ ไม่มีข้อมูล
            if df_es.empty:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                    'data_masking':[es],
                    'account_num':[None],
                    'event_type_id':[None],
                    'costcode':[cc],
                    'predicted_min':[None],
                    'actual_volume':[0],
                    'predicted_max':[None],
                    'is_nomaly':[False],
                    'status':[None],
                    'remark':["⚫ ไม่มีข้อมูลของเบอร์นี้ ย้อนหลัง 6 เดือน"],
                    'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method':["No Data"]
                })], ignore_index=True)
                continue

            account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns else None
            event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns else None
            costcode = cc if cc else df_es['costcode'].dropna().unique()[0]

            actual_volume = df_es[(df_es['start_date'] >= pd.to_datetime(predict_start_date)) & (df_es['start_date'] <= pd.to_datetime(predict_end_date))]['volume_monthly'].sum()
            prev_volume   = df_es[(df_es['start_date'] >= pd.to_datetime(predict_start_date) - relativedelta(months=1)) & (df_es['start_date'] <= pd.to_datetime(predict_end_date) - relativedelta(months=1))]['volume_monthly'].sum()
            prev_volume = prev_volume if prev_volume !=0 else None

            # 🔴 RULE 1: New Usage
            if (prev_volume is None or prev_volume==0) and actual_volume>0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                    'data_masking':[es],
                    'account_num':[account_num],
                    'event_type_id':[event_type_id],
                    'costcode':[costcode],
                    'predicted_min':[None],
                    'actual_volume':[actual_volume],
                    'predicted_max':[None],
                    'is_nomaly':[True],
                    'status':["New Usage"],
                    'remark':["❗ มี Usage ใหม่ (เดือนก่อน=0)"],
                    'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method':["Rule-based"]
                })], ignore_index=True)
                continue

            # =========================================
            # train model
            # =========================================
            df_train = df_es[(df_es['start_date'] >= train_start_date) & (df_es['start_date'] <= train_end_date)]
            df_train = df_train.groupby('start_date')['volume_monthly'].sum().reset_index()
            df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

            if df_train.shape[0] >= 6:
                model = Prophet()
                model.fit(df_train)
                forecast = model.predict(pd.DataFrame({'ds':[pd.to_datetime(predict_start_date)]}))
                predicted_min = max(0, forecast['yhat_lower'].values[0])
                predicted_max = max(predicted_min, forecast['yhat_upper'].values[0])
                method="Prophet"
            else:
                median_val = df_train['y'].median() if not df_train.empty else 1
                predicted_min = 0
                predicted_max = median_val
                method="Median fallback"
            if predicted_max < 1:
                predicted_max = max(df_train['y'].median(),1) if not df_train.empty else 1
                predicted_min = 0

            # 🔴 RULE 2: Drop to Zero
            if prev_volume is not None and prev_volume>0 and actual_volume==0:
                anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                    'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                    'data_masking':[es],
                    'account_num':[account_num],
                    'event_type_id':[event_type_id],
                    'costcode':[costcode],
                    'predicted_min':[predicted_min],
                    'actual_volume':[0],
                    'predicted_max':[predicted_max],
                    'is_nomaly':[True],
                    'status':["Drop to Zero"],
                    'remark':["❗ Usage หายไปจากเดือนก่อน"],
                    'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                    'method':[method]
                })], ignore_index=True)
                continue

            # =========================================
            # anomaly logic
            # =========================================
            diff = round((actual_volume - predicted_max)/predicted_max*100,2) if predicted_max else None
            if diff is None or abs(diff)<=5:
                is_nomaly=False
                status="Normal"
                remark=""
            else:
                is_nomaly=True
                if diff>50:
                    status="High"
                    remark="❗ เพิ่มขึ้นผิดปกติ เทียบ 6 เดือนย้อนหลัง"
                elif diff<-50:
                    status="Low"
                    remark="❗ ลดลงผิดปกติ เทียบ 6 เดือนย้อนหลัง"
                else:
                    status="Normal"
                    remark=""

            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                'data_masking':[es],
                'account_num':[account_num],
                'event_type_id':[event_type_id],
                'costcode':[costcode],
                'predicted_min':[predicted_min],
                'actual_volume':[actual_volume],
                'predicted_max':[predicted_max],
                'is_nomaly':[is_nomaly],
                'status':[status],
                'remark':[remark],
                'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],
                'method':[method]
            })], ignore_index=True)

        # =========================================
        # 7️⃣ Sort FALSE ก่อน
        # =========================================
        def remark_priority(x):
            if "❗" in str(x):
                return 0
            elif "🟡" in str(x):
                return 1
            elif "⚫" in str(x):
                return 2
            else:
                return 3

        anomaly_results['is_nomaly_sort'] = anomaly_results['is_nomaly'].apply(lambda x: 0 if x==False else 1)
        anomaly_results['remark_sort'] = anomaly_results['remark'].apply(remark_priority)
        anomaly_results = anomaly_results.sort_values(by=['is_nomaly_sort','remark_sort']).drop(columns=['is_nomaly_sort','remark_sort'])

        # =========================================
        # 8️⃣ Show table & download
        # =========================================
        st.subheader("✅ Anomaly Results")
        st.dataframe(anomaly_results)
        tz = pytz.timezone('Asia/Bangkok')
        now = datetime.now(tz)
        file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
        anomaly_results.to_excel(file_name,index=False)
        st.download_button("📥 Download Excel", file_name, data=open(file_name,'rb').read(), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
