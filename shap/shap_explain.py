"""SHAP explainability for the best model (notebook cell 107).

Run:  python shap/shap_explain.py
NOTE: this folder must NOT contain a file called shap.py or __init__.py, otherwise
it would shadow the real ``shap`` library.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.preprocessing import load_feature_store  # noqa: E402


def main():
    config.ensure_dirs()
    X_train_fs, X_test_fs, _, _ = load_feature_store()
    leaderboard = pd.read_csv(config.LEADERBOARD_PATH)
    best_model_name = leaderboard.iloc[0]["Model"]
    best_method = leaderboard.iloc[0]["Method"]
    from models.model_zoo import model_file_path
    winning_model = joblib.load(model_file_path(best_model_name, best_method))

    try:
        # Use a subset so SHAP remains fast
        shap_sample = X_test_fs.iloc[:min(300, len(X_test_fs))]
        if best_model_name in ["Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"]:
            explainer = shap.TreeExplainer(winning_model)
            shap_values = explainer.shap_values(shap_sample)
        else:
            explainer = shap.LinearExplainer(winning_model, X_train_fs)
            shap_values = explainer.shap_values(shap_sample)
        # Some versions/models return a list or a (n, features, classes) array
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]

        plt.figure()
        shap.summary_plot(shap_values, shap_sample, feature_names=X_test_fs.columns, show=False)
        plt.title(f"SHAP Summary - {best_model_name}")
        plt.tight_layout()
        plt.savefig(config.SHAP_PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values, shap_sample, feature_names=X_test_fs.columns, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {best_model_name}")
        plt.tight_layout()
        plt.savefig(config.SHAP_PLOTS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP plots saved in:", config.SHAP_PLOTS_DIR)
    except Exception as e:
        print("SHAP could not be generated for this model/environment.")
        print("Reason:", e)
        print("\nFeature importance from the previous cells can still be used.")


if __name__ == "__main__":
    main()
