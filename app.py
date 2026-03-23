import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tabulate import tabulate
import pytz
from io import BytesIO

st.title("CDR Anomaly Detection")

# =========================================
# 1️⃣ Upload file
# =========================================
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')

    # =========================================
    # 2️⃣ Input predict dates
    # =========================================
    col1, col2 = st.columns(2)
    with col1:
        predict_start_date = st.date_input("Predict Start Date", value=df['start_date'].min())
    with col2:
        predict_end_date = st.date_input("Predict End Date", value=df['start_date'].max())

    train_start_date = predict_start_date - relativedelta(months=7)
    train_end_date   = predict_end_date   - relativedelta(months=2)

    # =========================================
    # 3️⃣ Input event
    # =========================================
    user_input = st.text_input("data_masking:costcode list (comma-separated, costcode optional)")
    event_pairs = []
    for pair in user_input.split(','):
        if ':' in pair:
            es, cc = pair.split(':', 1)
            event_pairs.append((es.strip(), cc.strip()))
        else:
            event_pairs.append((pair.strip(), None))

    # =========================================
    # 4️⃣ Prepare result table
    # =========================================
    anomaly_results = pd.DataFrame(columns=[
        'predict_range','data_masking','account_num','event_type_id','costcode',
        'predicted_min','actual_volume','predicted_max',
        'results','diff','remark','train_range','method'
    ])

    # =========================================
    # 5️⃣ Loop per event
    # =========================================
    for es, cc in event_pairs:
        df_es = df[df['data_masking']==es]
        if cc:
            df_es = df_es[df_es['costcode']==cc]

        if df_es.empty:
            anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
                'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
                'data_masking':[es],'account_num':[None],'event_type_id':[None],
                'costcode':[cc],'predicted_min':[0],'actual_volume':[0],'predicted_max':[0],
                'results':[False],'diff':[None],'remark':["ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"],
                'train_range':[f"{train_start_date} ถึง {train_end_date}"],'method':["No Data"]
            })], ignore_index=True)
            continue

        account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns and not df_es['account_num'].dropna().empty else None
        event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns and not df_es['event_type_id'].dropna().empty else None
        costcode_val = cc if cc else df_es['costcode'].dropna().unique()[0]

        actual_volume = df_es[
            (df_es['start_date'] >= pd.to_datetime(predict_start_date)) &
            (df_es['start_date'] <= pd.to_datetime(predict_end_date))
        ]['volume_monthly'].sum()

        prev_volume = df_es[
            (df_es['start_date'] >= pd.to_datetime(predict_start_date) - relativedelta(months=1)) &
            (df_es['start_date'] <= pd.to_datetime(predict_end_date) - relativedelta(months=1))
        ]['volume_monthly'].sum()

        df_train = df_es[
            (df_es['start_date'] >= train_start_date) &
            (df_es['start_date'] <= train_end_date)
        ].groupby('start_date')['volume_monthly'].sum().reset_index()
        df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

        # forecast
        if df_train.shape[0]>=6:
            model = Prophet()
            model.fit(df_train)
            forecast = model.predict(pd.DataFrame({'ds':[predict_start_date]}))
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
        elif actual_volume<200:
            results_val = True
            diff_val = 0
            remark_txt="ปกติ (CDR < 200)"
        elif predicted_min <= actual_volume <= predicted_max:
            results_val = True
            diff_val = 0
            remark_txt="Diff แต่ยังอยู่ใน threshold"
        else:
            if actual_volume > predicted_max:
                diff_val = round((actual_volume - predicted_max)/predicted_max*100,2)
            else:
                diff_val = round((actual_volume - predicted_min)/predicted_min*100,2) if predicted_min>0 else 0
            abs_val = abs(diff_val)
            if 1<=actual_volume<=1_000:
                results_val = False if abs_val>=80 else True
            elif 1_001<=actual_volume<=100_000:
                results_val = False if abs_val>=50 else True
            elif 100_001<=actual_volume<=1_000_000:
                results_val = False if abs_val>=20 else True
            elif actual_volume>=1_000_001:
                results_val = False
            remark_txt = "❗ เพิ่มขึ้นผิดปกติ" if actual_volume>predicted_max else "❗ ลดลงผิดปกติ"

        anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
            'predict_range':[f"{predict_start_date} ถึง {predict_end_date}"],
            'data_masking':[es],
            'account_num':[account_num],
            'event_type_id':[event_type_id],
            'costcode':[costcode_val],
            'predicted_min':[predicted_min],
            'actual_volume':[actual_volume],
            'predicted_max':[predicted_max],
            'results':[results_val],
            'diff':[f"{diff_val}%" if diff_val is not None else None],
            'remark':[remark_txt],
            'train_range':[f"{train_start_date} ถึง {train_end_date}"],
            'method':[method_val]
        })], ignore_index=True)

    # =========================================
    # sort
    # =========================================
    anomaly_results['results_sort'] = anomaly_results['results'].apply(lambda x:1 if x else 0)
    anomaly_results = anomaly_results.sort_values(by=['results_sort','remark'], ascending=[True,True]).drop(columns=['results_sort'])

    # =========================================
    # show table
    # =========================================
    st.subheader("Anomaly Results")
    st.dataframe(anomaly_results)

    # download
    output = BytesIO()
    anomaly_results.to_excel(output,index=False)
    output.seek(0)
    st.download_button(
        label="Download Excel",
        data=output,
        file_name=f"cdr_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
