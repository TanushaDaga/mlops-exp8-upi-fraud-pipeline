"""ETL + Feature-store build.  Run:  python src/pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src import features as fe  # noqa: E402
from src import preprocessing as pp  # noqa: E402


def run_etl(track_mlflow=True):
    config.ensure_dirs()

    print("=" * 60, "\nEXTRACT\n", "=" * 60, sep="")
    df = dl.extract_raw()
    dl.inspect_dataset(df)
    df = dl.convert_types(df)
    print("\nMissing values:\n", dl.missing_value_summary(df))
    print("\nVALIDATION")
    dl.validate_data(df)
    df = dl.encode_target_and_binaries(df)
    print("\nOUTLIER SUMMARY (IQR)\n", dl.outlier_summary(df))
    dl.amount_outlier_fraud_rates(df)

    print("\n" + "=" * 60 + "\nSPLIT + FEATURE ENGINEERING\n" + "=" * 60)
    train_df, test_df = dl.split_data(df)
    stats = fe.compute_reference_stats(train_df)
    train_engineered = fe.engineer_features(train_df, stats)
    test_engineered = fe.engineer_features(test_df, stats)
    print("Training engineered shape:", train_engineered.shape)
    print("Testing engineered shape:", test_engineered.shape)
    print("Missing in train after cleaning:\n", train_engineered.isna().sum()[lambda s: s > 0])
    print("Missing in test after cleaning:\n", test_engineered.isna().sum()[lambda s: s > 0])
    print("\nEngineered features created:")
    for feature in fe.ENGINEERED_FEATURES:
        print(" ", feature)

    print("\n" + "=" * 60 + "\nPREPROCESSOR + FEATURE STORE\n" + "=" * 60)
    X_train_raw, X_test_raw, y_train, y_test = pp.prepare_xy(train_engineered, test_engineered)
    numeric_features, categorical_features = pp.identify_feature_types(X_train_raw)
    preprocessor = pp.build_preprocessor(numeric_features, categorical_features)
    X_train_processed, X_test_processed, feature_names = pp.fit_transform_preprocessor(
        preprocessor, X_train_raw, X_test_raw, y_train)
    print("Any missing in processed train?", X_train_processed.isna().any().any())
    print("Any missing in processed test?", X_test_processed.isna().any().any())
    print("Training class distribution:\n", y_train.value_counts())
    print("Testing class distribution:\n", y_test.value_counts())

    pp.save_feature_store(preprocessor, X_train_processed, y_train, X_test_processed, y_test, feature_names)
    joblib.dump(stats, config.REFERENCE_STATS_PATH)
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = pp.load_feature_store()
    print("\nFeature store successfully reloaded:", X_train_fs.shape, X_test_fs.shape)
    pp.verify_feature_store(X_train_fs, X_test_fs, y_train_fs, y_test_fs,
                            X_train_processed, X_test_processed, y_train, y_test)

    # Clean engineered dataset + raw test split (used by EDA / financial impact / tests)
    clean_engineered = pd.concat([train_engineered, test_engineered]).sort_index()
    clean_engineered.to_csv(config.CLEAN_ENGINEERED_PATH, index=False)
    test_df.to_csv(config.TEST_RAW_PATH, index=False)
    print("Clean engineered dataset saved at:", config.CLEAN_ENGINEERED_PATH)

    if track_mlflow:
        mlflow = config.setup_mlflow()
        with mlflow.start_run(run_name="FeatureStore_Build"):
            mlflow.set_tag("stage", "ETL_and_Feature_Store")
            mlflow.log_param("dataset", config.FILE_PATH)
            mlflow.log_param("dataset_rows", len(df))
            mlflow.log_param("dataset_columns", len(df.columns))
            mlflow.log_param("train_test_split", "80/20")
            mlflow.log_param("random_state", config.RANDOM_STATE)
            mlflow.log_param("feature_count", len(feature_names))
            mlflow.log_artifact(str(config.TRAIN_FEATURE_STORE_PATH), artifact_path="feature_store")
            mlflow.log_artifact(str(config.TEST_FEATURE_STORE_PATH), artifact_path="feature_store")
            mlflow.log_artifact(str(config.FEATURE_NAMES_PATH), artifact_path="feature_store")
            mlflow.log_artifact(str(config.PREPROCESSOR_PATH), artifact_path="feature_store")
        print("Feature-store build successfully tracked in MLflow.")
    return {"df": df, "train_df": train_df, "test_df": test_df,
            "train_engineered": train_engineered, "test_engineered": test_engineered}


if __name__ == "__main__":
    run_etl()
