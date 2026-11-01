"""All EDA graphs from the notebook, saved to eda/plots/.

Run:  python eda/eda_plots.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import features as fe  # noqa: E402

OUT = config.EDA_PLOTS_DIR


def _save(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()


def fraud_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Fraud_Flag")
    plt.title("Fraud vs Legitimate Transactions")
    plt.xlabel("Fraud Flag (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Count")
    _save("01_fraud_distribution.png")


def fraud_by_transaction_type(df):
    fraud_by_type = df.groupby("Transaction_Type")["Fraud_Flag"].agg(["count", "mean"]).sort_values("mean", ascending=False)
    fraud_by_type["Fraud_Rate_%"] = (fraud_by_type["mean"] * 100).round(2)
    print(fraud_by_type)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="Transaction_Type", y="Fraud_Flag")
    plt.title("Fraud Rate by Transaction Type")
    plt.ylabel("Fraud Rate")
    plt.xlabel("Transaction Type")
    plt.xticks(rotation=30)
    _save("02_fraud_by_transaction_type.png")


def amount_vs_fraud(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Fraud_Flag", y="Transaction_Amount")
    plt.title("Transaction Amount vs Fraud")
    plt.xlabel("Fraud Flag")
    plt.ylabel("Transaction Amount (Rs)")
    _save("03_transaction_amount_vs_fraud.png")


def fraud_by_hour(df):
    hourly_fraud = df.groupby("Transaction_Hour")["Fraud_Flag"].agg(["count", "mean"])
    hourly_fraud["Fraud_Rate_%"] = hourly_fraud["mean"] * 100
    plt.figure(figsize=(12, 5))
    sns.barplot(x=hourly_fraud.index, y=hourly_fraud["Fraud_Rate_%"])
    plt.title("Fraud Rate by Transaction Hour")
    plt.xlabel("Transaction Hour")
    plt.ylabel("Fraud Rate (%)")
    _save("04_fraud_by_hour.png")


def device_and_kyc(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=df, x="Device_Type", y="Fraud_Flag", ax=axes[0])
    axes[0].set_title("Fraud Rate by Device Type")
    axes[0].set_ylabel("Fraud Rate")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=df, x="KYC_Status", y="Fraud_Flag", ax=axes[1])
    axes[1].set_title("Fraud Rate by KYC Status")
    axes[1].set_ylabel("Fraud Rate")
    plt.tight_layout()
    _save("05_device_and_kyc.png")


def behavioral_heatmap(df):
    heatmap = df.pivot_table(values="Fraud_Flag", index="Login_Attempts", columns="Transactions_Last_24h", aggfunc="mean")
    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Fraud Rate: Login Attempts vs Transactions in Last 24h")
    plt.xlabel("Transactions Last 24h")
    plt.ylabel("Login Attempts")
    _save("06_behavioral_risk_heatmap.png")


def correlation_heatmap(df):
    train_df, _ = dl.split_data(df)
    stats = fe.compute_reference_stats(train_df)
    train_engineered = fe.engineer_features(train_df, stats)
    corr_matrix = train_engineered.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric and Engineered Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save("07_correlation_heatmap.png")


def main():
    config.ensure_dirs()
    df = dl.load_clean_dataframe()
    dl.amount_outlier_fraud_rates(df)
    print(dl.outlier_summary(df))
    fraud_distribution(df)
    fraud_by_transaction_type(df)
    amount_vs_fraud(df)
    fraud_by_hour(df)
    device_and_kyc(df)
    behavioral_heatmap(df)
    correlation_heatmap(df)
    print("EDA plots saved in:", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
