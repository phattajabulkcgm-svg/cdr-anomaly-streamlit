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

# =========================================
# 3️⃣ prepare
# =========================================
df['start_date'] = pd.to_datetime(df['start_date'], dayfirst=True, errors='coerce')

# =========================================
# 4️⃣ input predict
# =========================================
predict_start_date = pd.to_datetime(input("Predict Start Date (YYYY-MM-DD): "))
predict_end_date   = pd.to_datetime(input("Predict End Date (YYYY-MM-DD): "))

# train 6 เดือน + skip 1
train_start_date = predict_start_date - relativedelta(months=7)
train_end_date   = predict_end_date   - relativedelta(months=2)

# =========================================
# 5️⃣ input event
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
# 6️⃣ result table
# =========================================
anomaly_results = pd.DataFrame(columns=[
    'predict_range','data_masking','account_num','event_type_id','costcode',
    'predicted_min','actual_volume','predicted_max',
    'is_nomaly','diff','level','remark',
    'train_range','method'
])

# =========================================
# 7️⃣ loop
# =========================================
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
    # 🔴 RULE 1: New Usage
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
            'level': ["Normal"],
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
    ].groupby('start_date')['volume_monthly'].sum().reset_index()
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
    # 🔴 RULE 2: Drop to Zero
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
            'level': ["Normal"],
            'remark': ["❗ Usage หายไปจากเดือนก่อน"],
            'train_range': [f"{train_start_date.date()} ถึง {train_end_date.date()}"],
            'method': [method]
        })], ignore_index=True)
        continue

    # =========================================
    # anomaly logic ปรับใหม่ (TRUE/FALSE + remark เพิ่ม/ลด)
    # =========================================
    diff = round((actual_volume - predicted_max) / predicted_max * 100, 2) if predicted_max else None

    if diff is None:
        is_nomaly = False
        level = None
        remark = ""
    else:
        # threshold ตาม magnitude
        if actual_volume < 100:
            threshold = 5
        elif actual_volume <= 10_000:
            threshold = 10
        elif actual_volume <= 1_000_000:
            threshold = 30
        elif actual_volume <= 100_000_000:
            threshold = 50
        else:
            threshold = 80

        is_nomaly = abs(diff) > threshold
        level = "Normal"
        remark = ""
        if is_nomaly:
            if actual_volume - predicted_max > 0:
                remark = "❗ เพิ่มขึ้นผิดปกติ"
                if method == "Prophet":
                    remark += " เมื่อเทียบกับ 6 เดือนย้อนหลัง"
                else:
                    remark += " เมื่อเทียบกับเดือนที่ก่อนหน้า"
            else:
                remark = "❗ ลดลงผิดปกติ"
                if method == "Prophet":
                    remark += " เมื่อเทียบกับ 6 เดือนย้อนหลัง"
                else:
                    remark += " เมื่อเทียบกับเดือนที่ก่อนหน้า"

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

# =========================================
# sort (remark priority)
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

anomaly_results['is_nomaly_sort'] = anomaly_results['is_nomaly'].apply(lambda x: 1 if x else 0)
anomaly_results['remark_sort'] = anomaly_results['remark'].apply(remark_priority)

anomaly_results = anomaly_results.sort_values(
    by=['is_nomaly_sort','remark_sort'], ascending=False
).drop(columns=['is_nomaly_sort','remark_sort'])

# =========================================
# show
# =========================================
print(tabulate(anomaly_results, headers='keys', tablefmt='grid'))

# =========================================
# download
# =========================================
tz = pytz.timezone('Asia/Bangkok')
now = datetime.now(tz)
file_name = f"cdr_anomaly_{now.strftime('%Y%m%d_%H%M%S')}.xlsx"
anomaly_results.to_excel(file_name, index=False)
files.download(file_name)
