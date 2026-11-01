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

from src import ui  # noqa: E402
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
opts = meta["ui_options"]
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ form state + one-click scenarios
DEFAULTS = {
    "amount": 1500.0, "hour": 14, "ttype": "P2P", "merchant": opts["Merchant_Category"][0], "tdate": date(2024, 9, 11),
    "last24": 3, "age": 35, "acct": 1200, "kyc": "Verified", "city": opts["User_City"][0], "wallet": 20000.0,
    "bank": 200000.0, "prev": 300, "avg": 7000.0, "device": opts["Device_Type"][0], "appver": "v5.3.0",
    "login": 1, "devchg": False, "ipm": True, "susp": False, "rapid": False,
}
SCENARIOS = {
    "Everyday P2P payment": dict(DEFAULTS),
    "Late-night EMI spike": dict(DEFAULTS, amount=95000.0, hour=2, ttype="EMI", last24=20, age=29, acct=10, kyc="Not Submitted",
                                 wallet=500.0, bank=8000.0, prev=3, avg=3000.0, login=6, devchg=True, ipm=False, susp=True, rapid=True),
    "Borderline merchant payment": dict(DEFAULTS, amount=25000.0, hour=23, ttype="Merchant", last24=8, age=41, acct=90, kyc="Pending",
                                        wallet=6000.0, bank=40000.0, prev=25, avg=4000.0, login=3, devchg=True),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("history", [])


def load_scenario(name):
    for k, v in SCENARIOS[name].items():
        st.session_state[k] = v


def clear_history():
    st.session_state.history = []


# ------------------------------------------------------------------ page
st.markdown(ui.hero(meta), unsafe_allow_html=True)
tab_predict, tab_history, tab_how = st.tabs(["Check a transaction", "Session activity", "How it works"])

with tab_predict:
    st.markdown(ui.headline("Check a payment", "in one click", eyebrow="Live prediction",
                            sub="Pick a ready-made scenario or edit any field, then press the button to get the fraud risk."),
                unsafe_allow_html=True)
    cols = st.columns(len(SCENARIOS))
    for col, name in zip(cols, SCENARIOS):
        col.button(f"⚡ {name}", key=f"sc_{name}", on_click=load_scenario, args=(name,), use_container_width=True)

    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(ui.card_title("Transaction", "What is being paid"), unsafe_allow_html=True)
            amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=100.0, key="amount")
            hour = st.slider("Transaction Hour", 0, 23, key="hour")
            txn_type = st.selectbox("Transaction Type", opts["Transaction_Type"], key="ttype")
            merchant = st.selectbox("Merchant Category (Merchant txns only)", opts["Merchant_Category"], key="merchant")
            txn_date = st.date_input("Transaction Date", key="tdate")
            last_24h = st.number_input("Transactions in Last 24h", 0, 1000, key="last24")
        with c2:
            st.markdown(ui.card_title("User & Account", "Who is paying"), unsafe_allow_html=True)
            user_age = st.number_input("User Age", 0, 120, key="age")
            account_age = st.number_input("Account Age (days)", 0, 5000, key="acct")
            kyc = st.selectbox("KYC Status", list(KYC_MAP), key="kyc")
            city = st.selectbox("User City", opts["User_City"], key="city")
            wallet = st.number_input("Wallet Balance (₹)", min_value=0.0, step=500.0, key="wallet")
            bank = st.number_input("Bank Balance (₹)", min_value=0.0, step=1000.0, key="bank")
            prev_txn = st.number_input("Previous Transaction Count", 0, 100000, key="prev")
            avg_txn = st.number_input("Average Transaction Value (₹)", min_value=0.0, step=100.0, key="avg")
        with c3:
            st.markdown(ui.card_title("Device & Security", "How it is being paid"), unsafe_allow_html=True)
            device = st.selectbox("Device Type", opts["Device_Type"], key="device")
            app_ver = st.selectbox("App Version", opts["App_Version"], key="appver")
            login_attempts = st.number_input("Login Attempts", 0, 100, key="login")
            device_change = st.checkbox("Device changed recently", key="devchg")
            ip_match = st.checkbox("IP location matches usual location", key="ipm")
            suspicious = st.checkbox("Suspicious device flagged", key="susp")
            rapid = st.checkbox("Rapid transactions flagged", key="rapid")
        submitted = st.form_submit_button("🔍 Predict fraud risk", use_container_width=True)

    if submitted:
        metrics["requests"].inc()
        start = time.perf_counter()
        try:
            payload = {
                "Account_Age_Days": int(account_age), "User_Age": int(user_age), "KYC_Status": KYC_MAP[kyc],
                "Transaction_Amount": float(amount), "Transaction_Hour": int(hour),
                "Device_Change_Flag": int(device_change), "Previous_Transaction_Count": int(prev_txn),
                "Avg_Transaction_Value": float(avg_txn), "Transactions_Last_24h": int(last_24h),
                "Login_Attempts": int(login_attempts), "IP_Location_Match": int(ip_match),
                "Suspicious_Device": int(suspicious), "Rapid_Transaction_Flag": int(rapid),
                "Wallet_Balance": float(wallet), "Bank_Balance": float(bank), "User_City": city,
                "Transaction_Type": txn_type, "Merchant_Category": merchant if txn_type == "Merchant" else None,
                "Device_Type": device, "App_Version": app_ver, "Transaction_Date": txn_date,
            }
            result = predict(payload, bundle)
            metrics["latency"].observe(time.perf_counter() - start)
            metrics["predictions"].labels(predicted_class=result["label"]).inc()
            metrics["probability"].observe(result["probability"])

            left, right = st.columns([3, 2])
            with left:
                st.markdown(ui.result_card(result, float(amount)), unsafe_allow_html=True)
                st.caption(f"Threshold {result['threshold']:.3f} · latency {(time.perf_counter() - start) * 1000:.1f} ms")
            with right:
                st.markdown(ui.card_title("Warning signs", "Read from the details you entered"), unsafe_allow_html=True)
                st.markdown(ui.chips(ui.risk_signals(payload)), unsafe_allow_html=True)
            st.session_state.history.insert(0, {
                "Time": time.strftime("%H:%M:%S"), "Type": txn_type, "Amount (₹)": float(amount), "Hour": int(hour),
                "Fraud probability": round(result["probability"], 4), "Decision": result["label"]})
            st.session_state.history = st.session_state.history[:50]
        except InvalidInputError as exc:
            metrics["errors"].labels(error_type="invalid_input").inc()
            st.warning(f"Invalid input: {exc}")
        except Exception as exc:  # noqa: BLE001 - anything else is a server-side error
            metrics["errors"].labels(error_type="internal_error").inc()
            st.error(f"Prediction failed: {exc}")

with tab_history:
    st.markdown(ui.headline("Your session", "at a glance", eyebrow="Activity",
                            sub="Every prediction made in this browser session is listed here."), unsafe_allow_html=True)
    hist = st.session_state.history
    n_fraud = sum(1 for h in hist if h["Decision"] == "Fraud")
    flagged_amt = sum(h["Amount (₹)"] for h in hist if h["Decision"] == "Fraud")
    a, b, c, d = st.columns(4)
    a.metric("Predictions", len(hist))
    b.metric("Flagged as fraud", n_fraud)
    c.metric("Legitimate", len(hist) - n_fraud)
    d.metric("Value flagged (₹)", f"{flagged_amt:,.0f}")
    if hist:
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.button("Clear history", on_click=clear_history)
    else:
        st.info("No predictions yet - run one from the first tab.")

with tab_how:
    st.markdown(ui.headline("How the service", "makes a decision", eyebrow="Under the hood",
                            sub=f"{meta['best_model']} trained with {meta['training_method']} on 2,240 UPI transactions."),
                unsafe_allow_html=True)
    st.markdown(ui.how_it_works(meta), unsafe_allow_html=True)
    m = meta["metrics"]
    st.markdown(ui.headline("Model quality", "on unseen data", sub="Measured on the 560 held-out test transactions."),
                unsafe_allow_html=True)
    cols = st.columns(6)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Fraud", "ROC-AUC", "PR-AUC"]):
        col.metric(key, f"{m[key]:.3f}")

st.markdown(ui.powered_by(), unsafe_allow_html=True)
st.markdown(ui.footer(), unsafe_allow_html=True)
