"""Streamlit inference UI + Prometheus instrumentation.

* UI           : http://localhost:8501
* /metrics     : http://localhost:8000/metrics   (prometheus_client HTTP server)

Run locally:  streamlit run src/app.py
"""
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402
from prometheus_client import Counter, Gauge, Histogram, start_http_server  # noqa: E402

from src.inference import KYC_MAP, InvalidInputError, load_bundle, predict  # noqa: E402

METRICS_PORT = 8000


@st.cache_resource
def init_metrics():
    """Create the metrics ONCE per process and start the /metrics HTTP endpoint."""
    metrics = {
        "requests": Counter("ml_prediction_requests_total", "Total prediction requests received"),
        "predictions": Counter("ml_predictions_total", "Predictions by predicted class", ["predicted_class"]),
        "errors": Counter("ml_prediction_errors_total", "Failed prediction requests", ["error_type"]),
        "latency": Histogram("ml_prediction_latency_seconds", "Prediction latency in seconds",
                             buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)),
        "probability": Histogram("ml_fraud_probability", "Distribution of predicted fraud probability",
                                 buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        "up": Gauge("ml_app_up", "1 when the ML service is running"),
    }
    metrics["up"].set(1)
    for label in ("Fraud", "Legitimate"):  # expose zero-valued series from the start
        metrics["predictions"].labels(predicted_class=label)
    for err in ("invalid_input", "internal_error"):
        metrics["errors"].labels(error_type=err)
    try:
        start_http_server(METRICS_PORT)
    except OSError:  # port already bound (e.g. hot reload) - metrics server is already running
        pass
    return metrics


@st.cache_resource
def get_bundle():
    return load_bundle()


st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="wide")
metrics = init_metrics()
bundle = get_bundle()
meta = bundle["meta"]
ui = meta["ui_options"]

st.title("🛡️ UPI Fraud Detection Service")
st.caption(f"Model: **{meta['best_model']}** ({meta['training_method']}) · "
           f"decision threshold {meta['optimal_threshold']:.3f} · "
           f"ROC-AUC {meta['metrics']['ROC-AUC']:.3f} · F1-Macro {meta['metrics']['F1-Macro']:.3f}")

with st.form("txn_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Transaction")
        amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1500.0, step=100.0)
        hour = st.slider("Transaction Hour", 0, 23, 14)
        txn_type = st.selectbox("Transaction Type", ui["Transaction_Type"], index=ui["Transaction_Type"].index("P2P"))
        merchant = st.selectbox("Merchant Category (Merchant txns only)", ui["Merchant_Category"])
        txn_date = st.date_input("Transaction Date", value=date(2024, 9, 11))
        last_24h = st.number_input("Transactions in Last 24h", 0, 1000, 3)
    with c2:
        st.subheader("User & Account")
        user_age = st.number_input("User Age", 0, 120, 35)
        account_age = st.number_input("Account Age (days)", 0, 5000, 1200)
        kyc = st.selectbox("KYC Status", list(KYC_MAP), index=2)
        city = st.selectbox("User City", ui["User_City"])
        wallet = st.number_input("Wallet Balance (₹)", 0.0, value=20000.0, step=500.0)
        bank = st.number_input("Bank Balance (₹)", 0.0, value=200000.0, step=1000.0)
        prev_txn = st.number_input("Previous Transaction Count", 0, 100000, 300)
        avg_txn = st.number_input("Average Transaction Value (₹)", 0.0, value=7000.0, step=100.0)
    with c3:
        st.subheader("Device & Security")
        device = st.selectbox("Device Type", ui["Device_Type"])
        app_ver = st.selectbox("App Version", ui["App_Version"], index=ui["App_Version"].index("v5.3.0"))
        login_attempts = st.number_input("Login Attempts", 0, 100, 1)
        device_change = st.checkbox("Device changed recently")
        ip_match = st.checkbox("IP location matches usual location", value=True)
        suspicious = st.checkbox("Suspicious device flagged")
        rapid = st.checkbox("Rapid transactions flagged")
    submitted = st.form_submit_button("🔍 Predict fraud risk", use_container_width=True)

if submitted:
    metrics["requests"].inc()
    start = time.perf_counter()
    try:
        result = predict({
            "Account_Age_Days": int(account_age), "User_Age": int(user_age), "KYC_Status": KYC_MAP[kyc],
            "Transaction_Amount": float(amount), "Transaction_Hour": int(hour),
            "Device_Change_Flag": int(device_change), "Previous_Transaction_Count": int(prev_txn),
            "Avg_Transaction_Value": float(avg_txn), "Transactions_Last_24h": int(last_24h),
            "Login_Attempts": int(login_attempts), "IP_Location_Match": int(ip_match),
            "Suspicious_Device": int(suspicious), "Rapid_Transaction_Flag": int(rapid),
            "Wallet_Balance": float(wallet), "Bank_Balance": float(bank), "User_City": city,
            "Transaction_Type": txn_type, "Merchant_Category": merchant if txn_type == "Merchant" else None,
            "Device_Type": device, "App_Version": app_ver, "Transaction_Date": txn_date,
        }, bundle)
        metrics["latency"].observe(time.perf_counter() - start)
        metrics["predictions"].labels(predicted_class=result["label"]).inc()
        metrics["probability"].observe(result["probability"])
        if result["is_fraud"]:
            st.error(f"🚨 **FRAUD SUSPECTED** — fraud probability {result['probability']:.1%}")
        else:
            st.success(f"✅ **Legitimate transaction** — fraud probability {result['probability']:.1%}")
        st.progress(min(result["probability"], 1.0))
        st.caption(f"Threshold {result['threshold']:.3f} · latency {(time.perf_counter() - start) * 1000:.1f} ms")
    except InvalidInputError as exc:
        metrics["errors"].labels(error_type="invalid_input").inc()
        st.warning(f"Invalid input: {exc}")
    except Exception as exc:  # noqa: BLE001 - anything else is a server-side error
        metrics["errors"].labels(error_type="internal_error").inc()
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Prometheus metrics are exposed at port 8000 (`/metrics`).")
