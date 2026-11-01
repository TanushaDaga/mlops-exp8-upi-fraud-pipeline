"""Train, evaluate and track all 7 models x 2 balancing methods in MLflow.

Run:  python models/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

from models.model_zoo import (MODEL_PARAMETERS, apply_smote, calculate_scale_pos_weight,  # noqa: E402
                              create_models, model_file_path)
from src.preprocessing import load_feature_store  # noqa: E402


def train_all():
    config.ensure_dirs()
    mlflow = config.setup_mlflow()
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    print("MLflow Experiment:", config.EXPERIMENT_NAME)

    X_train_fs, X_test_fs, y_train_fs, y_test_fs = load_feature_store()
    print("\nHandle class imbalance using SMOTE")
    X_train_smote, y_train_smote = apply_smote(X_train_fs, y_train_fs)
    print("\nCalculate class weight / scale weight")
    scale_pos_weight = calculate_scale_pos_weight(y_train_fs)

    results = []
    for method in config.METHODS:
        print("\nMETHOD:", method)
        if method == "Class Weights":
            X_train_current, y_train_current = X_train_fs, y_train_fs
        else:
            X_train_current, y_train_current = X_train_smote, y_train_smote
        models = create_models(method, scale_pos_weight)

        for model_name, model in models.items():
            print(f"\nTraining {model_name} using {method}...")
            model.fit(X_train_current, y_train_current)

            probabilities = model.predict_proba(X_test_fs)[:, 1]
            predictions = (probabilities >= config.DEFAULT_THRESHOLD).astype(int)

            accuracy = accuracy_score(y_test_fs, predictions)
            precision = precision_score(y_test_fs, predictions, zero_division=0)
            recall = recall_score(y_test_fs, predictions, zero_division=0)
            f1_fraud = f1_score(y_test_fs, predictions, zero_division=0)
            f1_macro = f1_score(y_test_fs, predictions, average="macro", zero_division=0)
            roc_auc = roc_auc_score(y_test_fs, probabilities)
            pr_auc = average_precision_score(y_test_fs, probabilities)

            results.append({"Model": model_name, "Method": method, "Accuracy": accuracy,
                            "Precision": precision, "Recall": recall, "F1-Fraud": f1_fraud,
                            "F1-Macro": f1_macro, "ROC-AUC": roc_auc, "PR-AUC": pr_auc})

            # ---------------------------------------------------- MLFLOW RUN
            run_name = model_name.replace(" ", "_") + "_" + method.replace(" ", "_")
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("imbalance_method", method)
                mlflow.set_tag("dataset", "UPI Fraud Dataset")
                mlflow.log_param("model", model_name)
                mlflow.log_param("imbalance_method", method)
                mlflow.log_param("test_size", config.TEST_SIZE)
                mlflow.log_param("random_state", config.RANDOM_STATE)
                mlflow.log_param("feature_count", X_train_fs.shape[1])
                for param_name, param_value in MODEL_PARAMETERS[model_name].items():
                    mlflow.log_param(param_name, param_value)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_fraud", f1_fraud)
                mlflow.log_metric("f1_macro", f1_macro)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("pr_auc", pr_auc)

                model_path = model_file_path(model_name, method)
                joblib.dump(model, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                mlflow.log_artifact(str(config.TRAIN_FEATURE_STORE_PATH), artifact_path="feature_store")
                print(f"Accuracy={accuracy:.4f} | F1={f1_fraud:.4f} | F1-Macro={f1_macro:.4f} | ROC-AUC={roc_auc:.4f}")

    pd.DataFrame(results).to_csv(config.RAW_RESULTS_PATH, index=False)
    print("\nAll model experiments completed.")
    return results


if __name__ == "__main__":
    train_all()
