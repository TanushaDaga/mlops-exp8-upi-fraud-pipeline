"""Central configuration for the UPI Fraud Detection MLOps pipeline.

NOTE: thread settings must be applied before numpy / sklearn are imported,
so every entry-point script imports ``src.config`` first.
"""
import os
import sys
import warnings
from pathlib import Path

# Thread settings for stable execution on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- data
RAW_DATA_PATH = ROOT / "data" / "raw" / "PS2 - DataSet.xlsx"
FILE_PATH = "PS2 - DataSet.xlsx"
SHEET_NAME = "UPI_Fraud_Transactions"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_STORE_DIR = ROOT / "data" / "feature_store"

PREPROCESSOR_PATH = FEATURE_STORE_DIR / "preprocessor.joblib"
TRAIN_FEATURE_STORE_PATH = FEATURE_STORE_DIR / "upi_train_features.csv"
TEST_FEATURE_STORE_PATH = FEATURE_STORE_DIR / "upi_test_features.csv"
FEATURE_NAMES_PATH = FEATURE_STORE_DIR / "feature_names.csv"
REFERENCE_STATS_PATH = FEATURE_STORE_DIR / "reference_stats.joblib"
CLEAN_ENGINEERED_PATH = PROCESSED_DIR / "upi_clean_engineered_data.csv"
TEST_RAW_PATH = PROCESSED_DIR / "test_raw.csv"

# ----------------------------------------------------------- artifacts
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SERVING_DIR = ARTIFACTS_DIR / "serving"
RAW_RESULTS_PATH = ARTIFACTS_DIR / "training_results.csv"
LEADERBOARD_PATH = ARTIFACTS_DIR / "model_leaderboard.csv"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
FINAL_REPORT_PATH = ARTIFACTS_DIR / "final_model_report.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"

EDA_PLOTS_DIR = ROOT / "eda" / "plots"
EVAL_PLOTS_DIR = ROOT / "evaluation" / "plots"
SHAP_PLOTS_DIR = ROOT / "shap" / "plots"

# -------------------------------------------------------------- mlflow
MLFLOW_TRACKING_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
EXPERIMENT_NAME = "UPI_Fraud_ETL_FeatureStore_Experiment"
REGISTERED_MODEL_NAME = "UPI_Fraud_Best_Model"

# ------------------------------------------------------------ modelling
TARGET = "Fraud_Flag"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
METHODS = ["Class Weights", "SMOTE"]


def ensure_dirs():
    for d in (PROCESSED_DIR, FEATURE_STORE_DIR, ARTIFACTS_DIR, MODELS_DIR,
              SERVING_DIR, EDA_PLOTS_DIR, EVAL_PLOTS_DIR, SHAP_PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_mlflow():
    """Use a database backend so MLflow UI and scripts share the SAME store."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow
