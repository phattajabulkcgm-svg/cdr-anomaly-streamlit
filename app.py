# =========================================
# 0️⃣ install
# =========================================
!pip uninstall prophet pystan cmdstanpy -y -q
!pip install prophet==1.2.1 cmdstanpy==1.1.0 -q

# =========================================
# 1️⃣ import
# =========================================
import pandas as pd
from prophet import Prophet
from tabulate import tabulate
from datetime import datetime
from dateutil.relativedelta import relativedelta
from google.colab import files
import pytz

# =========================================
# 2️⃣ upload
# =========================================
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower()
df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')

# =========================================
# 3️⃣ input predict
# =========================================
predict_start_date = pd.to_datetime(input("Predict Start Date (YYYY-MM-DD): "))
predict_end_date   = pd.to_datetime(input("Predict End Date (YYYY-MM-DD): "))

# train 6 เดือน + skip 1 เดือน
train_start_date = predict_start_date - relativedelta(months=7)
train_end_date   = predict_end_date   - relativedelta(months=2)

# =========================================
# 4️⃣ input event
# =========================================
user_input = input("data_masking:costcode list (comma-separated, costcode optional): ")
event_pairs = []
for pair in user_input.split(','):
    if ':' in pair:
        es, cc = pair.split(':', 1)
        event_pairs.append((es.strip(), cc.strip()))
    else:
        event_pairs.append((pair.strip(), None))

# =========================================
# 5️⃣ result table
# =========================================
anomaly_results = pd.DataFrame(columns=[
    'predict_range','data_masking','account_num','event_type_id','costcode',
    'predicted_min','actual_volume','predicted_max',
    'results','diff','remark','train_range','method'
])

# =========================================
# 6️⃣ loop per event
# =========================================
for es, cc in event_pairs:
    df_es = df[df['data_masking'] == es]
    if cc:
        df_es = df_es[df_es['costcode'] == cc]

    if df_es.empty:
        anomaly_results = pd.concat([anomaly_results, pd.DataFrame({
            'predict_range':[f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
            'data_masking':[es],'account_num':[None],'event_type_id':[None],
            'costcode':[cc],'predicted_min':[0],'actual_volume':[0],'predicted_max':[0],
            'results':[False],'diff':[None],'remark':["ไม่มีเคยมี CDR ใน 6 เดือนล่าสุด"],
            'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],'method':["No Data"]
        })], ignore_index=True)
        continue

    account_num = df_es['account_num'].dropna().unique()[0] if 'account_num' in df_es.columns and not df_es['account_num'].dropna().empty else None
    event_type_id = df_es['event_type_id'].dropna().unique()[0] if 'event_type_id' in df_es.columns and not df_es['event_type_id'].dropna().empty else None
    costcode_val = cc if cc else df_es['costcode'].dropna().unique()[0]

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

    df_train = df_es[
        (df_es['start_date'] >= train_start_date) &
        (df_es['start_date'] <= train_end_date)
    ].groupby('start_date')['volume_monthly'].sum().reset_index()
    df_train.rename(columns={'start_date':'ds','volume_monthly':'y'}, inplace=True)

    # =========================
    # train forecast
    # =========================
    if df_train.shape[0] >= 6:
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

    # =========================
    # rules & results
    # =========================
    # ถ้าเดือนก่อน =0 แล้วเดือนนี้ >0 → เพิ่มขึ้น 100%
    if prev_volume==0 and actual_volume>0:
        results_val = False
        diff_val = 100
        remark_txt = "❗ มี Usage ใหม่ (เดือนก่อน = 0)"
    # ถ้าเดือนนี้ =0 → FALSE
    elif actual_volume==0:
        if prev_volume==0:
            remark_txt="❗ Usage 0 ใน 2 เดือนล่าสุด"
        else:
            remark_txt="❗ Usage หายไปจากเดือนก่อน"
        results_val = False
        diff_val = -100
    # CDR น้อยกว่า 200 → ปกติ
    elif actual_volume < 200:
        results_val = True
        diff_val = 0
        remark_txt = "ปกติ (CDR < 200)"
    # อยู่ใน min–max → TRUE
    elif predicted_min <= actual_volume <= predicted_max:
        results_val = True
        diff_val = 0
        remark_txt = "Diff แต่ยังอยู่ใน threshold"
    # อยู่นอก min–max → คำนวณ % diff
    else:
        if actual_volume > predicted_max:
            diff_val = round((actual_volume - predicted_max)/predicted_max*100,2)
        else:
            diff_val = round((actual_volume - predicted_min)/predicted_min*100,2) if predicted_min>0 else 0
        # threshold ตาม volume
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
        'predict_range':[f"{predict_start_date.date()} ถึง {predict_end_date.date()}"],
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
        'train_range':[f"{train_start_date.date()} ถึง {train_end_date.date()}"],
        'method':[method_val]
    })], ignore_index=True)

# =========================================
# sort by results & remark
# =========================================
anomaly_results['results_sort'] = anomaly_results['results'].apply(lambda x: 1 if x else 0)
anomaly_results = anomaly_results.sort_values(by=['results_sort','remark'], ascending=[True,True]).drop(columns=['results_sort'])

# =========================================
# show & download
# =========================================
print(tabulate(anomaly_results, headers='keys', tablefmt='grid'))

tz = pytz.timezone('Asia/Bangkok')
now = datetime.now(tz)
file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
anomaly_results.to_excel(file_name,index=False)
files.download(file_name)
