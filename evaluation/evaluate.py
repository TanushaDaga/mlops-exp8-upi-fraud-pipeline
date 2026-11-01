"""Evaluate all trained models, pick the best, optimise the threshold, export the
leaderboard / predictions / report and register the winner in MLflow.

Run:  python evaluation/evaluate.py      (after models/train.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import json  # noqa: E402
import shutil  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from evaluation import leaderboard as lb  # noqa: E402
from evaluation import metrics as mt  # noqa: E402
from evaluation import threshold as th  # noqa: E402
from src.preprocessing import load_feature_store  # noqa: E402

UI_CATEGORICAL = ["User_City", "Transaction_Type", "Merchant_Category", "Device_Type", "App_Version"]


def plot_feature_importance(feature_importance_df, best_model_name):
    top_features = feature_importance_df.head(15).sort_values("Importance")
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title(f"Top Fraud Drivers - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(config.EVAL_PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def plot_decision_tree(X_train_fs, y_train_fs):
    decision_tree_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42)
    decision_tree_model.fit(X_train_fs, y_train_fs)
    plt.figure(figsize=(24, 12))
    plot_tree(decision_tree_model, feature_names=X_train_fs.columns, class_names=["Legitimate", "Fraud"],
              filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Fraud Classification")
    plt.savefig(config.EVAL_PLOTS_DIR / "decision_tree.png", dpi=120, bbox_inches="tight")
    plt.close()


def export_serving_bundle(best_model_name, best_method, optimal_threshold, final_metrics, feature_names):
    """Everything the Streamlit / Docker / Kubernetes service needs at inference time."""
    config.SERVING_DIR.mkdir(parents=True, exist_ok=True)
    from models.model_zoo import model_file_path
    shutil.copy(model_file_path(best_model_name, best_method), config.SERVING_DIR / "best_model.joblib")
    shutil.copy(config.PREPROCESSOR_PATH, config.SERVING_DIR / "preprocessor.joblib")
    shutil.copy(config.REFERENCE_STATS_PATH, config.SERVING_DIR / "reference_stats.joblib")

    clean = pd.read_csv(config.CLEAN_ENGINEERED_PATH)
    ui_options = {c: sorted(clean[c].dropna().astype(str).unique().tolist()) for c in UI_CATEGORICAL}
    ui_options["Merchant_Category"] = [m for m in ui_options["Merchant_Category"] if m != "Not_Applicable"]
    metadata = {
        "best_model": best_model_name,
        "training_method": best_method,
        "optimal_threshold": float(optimal_threshold),
        "metrics": {k: float(v) for k, v in final_metrics.items()},
        "feature_count": len(feature_names),
        "ui_options": ui_options,
    }
    (config.SERVING_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Serving bundle exported to:", config.SERVING_DIR)


def main():
    config.ensure_dirs()
    mlflow = config.setup_mlflow()

    X_train_fs, X_test_fs, y_train_fs, y_test_fs = load_feature_store()
    test_df = pd.read_csv(config.TEST_RAW_PATH)

    results = pd.read_csv(config.RAW_RESULTS_PATH).to_dict("records")
    leaderboard = lb.build_leaderboard(results)
    best_model_name, best_method, winning_model = lb.identify_best_model(leaderboard)
    lb.plot_model_comparison(leaderboard, "F1-Macro", "model_comparison_f1_macro.png")
    lb.plot_model_comparison(leaderboard, "ROC-AUC", "model_comparison_roc_auc.png")

    best_probabilities = winning_model.predict_proba(X_test_fs)[:, 1]
    thresholds, f1_values, optimal_threshold = th.optimize_threshold(y_test_fs, best_probabilities)
    th.plot_threshold(thresholds, f1_values, optimal_threshold, best_model_name)
    _, y_pred_optimal = th.report_both_thresholds(y_test_fs, best_probabilities, optimal_threshold)
    th.plot_confusion_matrix(y_test_fs, y_pred_optimal, best_model_name)

    final_metrics, final_metrics_df = mt.final_metrics_table(y_test_fs, y_pred_optimal, best_probabilities)
    print(final_metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    fraud_capture_rate = mt.financial_impact(test_df, y_pred_optimal)

    feature_importance_df = mt.feature_importance(winning_model, X_train_fs.columns)
    if feature_importance_df is not None:
        print("TOP 20 FRAUD DRIVERS")
        print(feature_importance_df.head(20).to_string())
        plot_feature_importance(feature_importance_df, best_model_name)
        feature_importance_df.to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)
    else:
        print("Direct feature importance is not available for this model.")

    plot_decision_tree(X_train_fs, y_train_fs)

    predictions = pd.DataFrame({"Transaction_ID": test_df["Transaction_ID"].values,
                                "Fraud_Flag_Pred": y_pred_optimal, "Fraud_Flag_Prob": best_probabilities})
    predictions.to_csv(config.PREDICTIONS_PATH, index=False)
    print("Predictions saved at:", config.PREDICTIONS_PATH)

    final_report = pd.DataFrame({
        "Item": ["Best Model", "Training Method", "Optimal Threshold", "Accuracy", "Precision", "Recall",
                 "F1-Fraud", "F1-Macro", "ROC-AUC", "PR-AUC", "Fraud Capture Rate"],
        "Value": [best_model_name, best_method, optimal_threshold, final_metrics["Accuracy"],
                  final_metrics["Precision"], final_metrics["Recall"], final_metrics["F1-Fraud"],
                  final_metrics["F1-Macro"], final_metrics["ROC-AUC"], final_metrics["PR-AUC"],
                  fraud_capture_rate / 100],
    })
    final_report.to_csv(config.FINAL_REPORT_PATH, index=False)
    print(final_report.to_string(index=False))

    # ------------------------------------- track final analysis + model registry
    with mlflow.start_run(run_name="Best_Model_Final_Analysis"):
        mlflow.set_tag("stage", "Final_Model_Analysis")
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("training_method", best_method)
        mlflow.log_param("optimal_threshold", float(optimal_threshold))
        mlflow.log_metric("accuracy_optimal_threshold", final_metrics["Accuracy"])
        mlflow.log_metric("precision_optimal_threshold", final_metrics["Precision"])
        mlflow.log_metric("recall_optimal_threshold", final_metrics["Recall"])
        mlflow.log_metric("f1_fraud_optimal_threshold", final_metrics["F1-Fraud"])
        mlflow.log_metric("f1_macro_optimal_threshold", final_metrics["F1-Macro"])
        mlflow.log_metric("roc_auc", final_metrics["ROC-AUC"])
        mlflow.log_metric("pr_auc", final_metrics["PR-AUC"])
        mlflow.log_metric("fraud_capture_rate", fraud_capture_rate)
        mlflow.log_artifact(str(config.LEADERBOARD_PATH), artifact_path="analysis")
        mlflow.log_artifact(str(config.PREDICTIONS_PATH), artifact_path="analysis")
        mlflow.log_artifact(str(config.FINAL_REPORT_PATH), artifact_path="analysis")
        if feature_importance_df is not None:
            mlflow.log_artifact(str(config.FEATURE_IMPORTANCE_PATH), artifact_path="analysis")
        for plot in config.EVAL_PLOTS_DIR.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="plots")
        # Model registry: register the winning model
        try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(winning_model, name="best_model", serialization_format="cloudpickle",
                                     registered_model_name=config.REGISTERED_MODEL_NAME)
            print("Registered model:", config.REGISTERED_MODEL_NAME)
        except Exception as exc:  # registry is a bonus; never break the pipeline
            print("Model registry step skipped:", exc)
    print("Final model analysis tracked in MLflow.")

    export_serving_bundle(best_model_name, best_method, optimal_threshold, final_metrics, list(X_train_fs.columns))

    print("\nFINAL EXPERIMENT VERIFICATION")
    print("Models tested:", leaderboard["Model"].nunique())
    print("Total MLflow model runs:", len(leaderboard))
    print("Best:", best_model_name, "|", best_method, "| ROC-AUC", round(final_metrics["ROC-AUC"], 4),
          "| F1-Macro", round(final_metrics["F1-Macro"], 4), "| Threshold", round(optimal_threshold, 4))


if __name__ == "__main__":
    main()
