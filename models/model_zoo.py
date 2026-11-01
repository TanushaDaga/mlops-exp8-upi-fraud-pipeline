"""Model definitions + class-imbalance handling (notebook cells 71-77).

Balancing techniques:
    * "Class Weights" - class_weight / scale_pos_weight / auto_class_weights
    * "SMOTE"         - synthetic minority over-sampling on the training set
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402,F401

from catboost import CatBoostClassifier  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402


def apply_smote(X_train_fs, y_train_fs):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_fs, y_train_fs)
    print("Before SMOTE:")
    print(y_train_fs.value_counts())
    print("\nAfter SMOTE:")
    print(y_train_smote.value_counts())
    print("\nOriginal training rows:", len(X_train_fs))
    print("SMOTE training rows:", len(X_train_smote))
    return X_train_smote, y_train_smote


def calculate_scale_pos_weight(y_train_fs):
    negative_count = (y_train_fs == 0).sum()
    positive_count = (y_train_fs == 1).sum()
    scale_pos_weight = negative_count / positive_count
    print("Legitimate transactions:", negative_count)
    print("Fraud transactions:", positive_count)
    print("XGBoost scale_pos_weight:", round(scale_pos_weight, 3))
    return scale_pos_weight


def create_models(method, scale_pos_weight):
    if method == "Class Weights":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss", n_jobs=1),
            "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       class_weight="balanced", random_state=42, verbosity=-1, n_jobs=1),
            "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, auto_class_weights="Balanced",
                                           random_seed=42, verbose=0, thread_count=1, allow_writing_files=False),
            "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, eval_metric="logloss", n_jobs=1),
            "LightGBM": LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbosity=-1, n_jobs=1),
            "CatBoost": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_seed=42, verbose=0, thread_count=1, allow_writing_files=False),
            "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
        }
    return models


# Model parameter dictionary for MLflow
MODEL_PARAMETERS = {
    "Logistic Regression": {"max_iter": 1000},
    "Decision Tree": {"max_depth": 8, "min_samples_leaf": 5},
    "Random Forest": {"n_estimators": 300, "min_samples_leaf": 2},
    "XGBoost": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "LightGBM": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
    "CatBoost": {"iterations": 300, "depth": 6, "learning_rate": 0.05},
    "AdaBoost": {"n_estimators": 300, "learning_rate": 0.05},
}


def model_file_path(model_name, method):
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    return config.MODELS_DIR / f"{safe_model_name}_{method.replace(' ', '_')}.joblib"
