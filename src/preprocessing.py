"""Preprocessor + feature store (notebook cells 47-67)."""
from src import config

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS_TO_DROP = [config.TARGET, "User_ID", "Transaction_ID", "Transaction_Date", "Transaction_Time", "Operating_System"]


def prepare_xy(train_engineered, test_engineered):
    X_train_raw = train_engineered.drop(columns=COLUMNS_TO_DROP)
    X_test_raw = test_engineered.drop(columns=COLUMNS_TO_DROP)
    y_train = train_engineered[config.TARGET].astype(int)
    y_test = test_engineered[config.TARGET].astype(int)
    print("X_train shape:", X_train_raw.shape)
    print("X_test shape:", X_test_raw.shape)
    print("\ny_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train_raw, X_test_raw, y_train, y_test


def identify_feature_types(X_train_raw):
    categorical_features = X_train_raw.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [c for c in X_train_raw.columns if c not in categorical_features]
    print("Numerical features:")
    print(numeric_features)
    print("\nCategorical features:")
    print(categorical_features)
    print("\nNumber of numerical features:", len(numeric_features))
    print("Number of categorical features:", len(categorical_features))
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    # Compatibility with different scikit-learn versions
    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", one_hot_encoder)])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def fit_transform_preprocessor(preprocessor, X_train_raw, X_test_raw, y_train):
    preprocessor.fit(X_train_raw, y_train)
    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)
    print("Total final model features:", len(feature_names))
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train_raw.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test_raw.index)
    return X_train_processed, X_test_processed, feature_names


def save_feature_store(preprocessor, X_train_processed, y_train, X_test_processed, y_test, feature_names):
    """Create the feature store (train/test CSVs, feature names, preprocessor)."""
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    print("Preprocessor saved at:", config.PREPROCESSOR_PATH)

    train_fs = X_train_processed.copy()
    train_fs[config.TARGET] = y_train
    test_fs = X_test_processed.copy()
    test_fs[config.TARGET] = y_test
    train_fs.to_csv(config.TRAIN_FEATURE_STORE_PATH, index=False)
    test_fs.to_csv(config.TEST_FEATURE_STORE_PATH, index=False)
    pd.DataFrame({"Feature_Name": feature_names}).to_csv(config.FEATURE_NAMES_PATH, index=False)
    print("Feature store created successfully.")


def load_feature_store():
    """Reload the feature store (models never touch raw data again)."""
    stored_train = pd.read_csv(config.TRAIN_FEATURE_STORE_PATH)
    stored_test = pd.read_csv(config.TEST_FEATURE_STORE_PATH)
    X_train_fs = stored_train.drop(columns=[config.TARGET])
    X_test_fs = stored_test.drop(columns=[config.TARGET])
    y_train_fs = stored_train[config.TARGET].astype(int)
    y_test_fs = stored_test[config.TARGET].astype(int)
    return X_train_fs, X_test_fs, y_train_fs, y_test_fs


def verify_feature_store(X_train_fs, X_test_fs, y_train_fs, y_test_fs,
                         X_train_processed, X_test_processed, y_train, y_test):
    checks = {
        "Training feature store matches original": np.allclose(X_train_fs.values, X_train_processed.values),
        "Testing feature store matches original": np.allclose(X_test_fs.values, X_test_processed.values),
        "Training target matches": np.array_equal(y_train_fs.values, y_train.values),
        "Testing target matches": np.array_equal(y_test_fs.values, y_test.values),
    }
    for name, ok in checks.items():
        print(f"{name}:", ok)
    return checks
