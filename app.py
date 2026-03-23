
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from dateutil.relativedelta import relativedelta

try:
    from prophet import Prophet
    PROPHET = True
except:
    PROPHET = False

st.title("📊 CDR Anomaly Detection App")

st.write("Upload Excel file with columns: start_date, volume_monthly, data_masking, costcode")

file = st.file_uploader("Upload Excel", type=["xlsx"])
predict_start = st.date_input("Predict start date")
predict_end   = st.date_input("Predict end date")

event_input = st.text_input(
    "Data masking (comma separated, e.g. 12345:CC01,67890)"
)

run_btn = st.button("🚀 Run anomaly detection")

def prepare_event_pairs(text):
    pairs = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            es, cc = raw.split(":", 1)
            pairs.append((es.strip(), cc.strip()))
        else:
            pairs.append((raw, None))
    return pairs

if run_btn:
    if not file:
        st.error("Please upload an Excel file!")
        st.stop()

    df = pd.read_excel(file)
    df.columns = df.columns.str.lower().str.strip()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    event_pairs = prepare_event_pairs(event_input)
    results = []

    months = pd.date_range(predict_start, predict_end, freq="MS")

    for es, cc in event_pairs:
        d = df[df["data_masking"] == es]
        if cc:
            d = d[d["costcode"] == cc]

        if d.empty:
            results.append({
                "data_masking": es,
                "costcode": cc,
                "remark": "No history"
            })
            continue

        actual = d[
            (d["start_date"] >= pd.to_datetime(predict_start)) &
            (d["start_date"] <= pd.to_datetime(predict_end))
        ]["volume_monthly"].sum()

        prev = d[
            (d["start_date"] >= pd.to_datetime(predict_start) - relativedelta(months=1)) &
            (d["start_date"] <= pd.to_datetime(predict_end) - relativedelta(months=1))
        ]["volume_monthly"].sum()

        prev = prev if prev != 0 else None

        if (prev is None or prev == 0) and actual > 0:
            results.append({
                "data_masking": es,
                "costcode": cc,
                "actual": actual,
                "remark": "❗ NEW usage"
            })
            continue

        # Train
        train_start = pd.to_datetime(predict_start) - relativedelta(months=7)
        train_end   = pd.to_datetime(predict_end) - relativedelta(months=2)

        df_train = d[
            (d["start_date"] >= train_start) &
            (d["start_date"] <= train_end)
        ].groupby("start_date", as_index=False)["volume_monthly"].sum()

        if PROPHET and df_train.shape[0] >= 6:
            try:
                m = Prophet()
                dfp = df_train.rename(columns={"start_date":"ds","volume_monthly":"y"})
                m.fit(dfp)

                future = pd.DataFrame({"ds": months})
                fc = m.predict(future)

                pred_max = fc["yhat_upper"].clip(lower=1).sum()

            except:
                pred_max = df_train["volume_monthly"].median() * len(months)
        else:
            pred_max = df_train["volume_monthly"].median() * len(months)

        diff = round((actual - pred_max)/pred_max*100,2) if pred_max else None

        results.append({
            "data_masking": es,
            "costcode": cc,
            "actual": actual,
            "predicted": pred_max,
            "diff_percent": diff
        })

    df_res = pd.DataFrame(results)
    st.dataframe(df_res)

    # Download Excel
    towrite = BytesIO()
    df_res.to_excel(towrite, index=False)
    towrite.seek(0)

    st.download_button(
        "📥 Download results",
        towrite,
        file_name="anomaly_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
