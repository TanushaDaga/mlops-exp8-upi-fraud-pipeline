# UPI Fraud Detection – End-to-End MLOps Pipeline (Experiment 8)

A complete machine-learning project (UPI transaction fraud detection) taken through the whole MLOps
lifecycle: **Git/GitHub → GitHub Actions CI/CD → MLflow tracking & registry → Docker / Compose →
Kubernetes (Minikube) → Prometheus + Grafana → Streamlit inference service.**

```
mlops-exp8-pipeline/
├── .github/workflows/ci-cd.yaml   # lint -> pytest -> docker build
├── data/
│   ├── raw/                       # PS2 - DataSet.xlsx (source data)
│   ├── processed/                 # clean engineered dataset
│   └── feature_store/             # train/test feature CSVs + fitted preprocessor
├── eda/                           # eda_plots.py -> eda/plots/*.png
├── src/                           # config, data_loader, features, preprocessing, pipeline, inference, app.py (Streamlit)
├── models/                        # model_zoo.py (7 models, Class Weights + SMOTE), train.py (MLflow)
├── evaluation/                    # leaderboard, threshold optimisation, confusion matrix, financial impact
├── shap/                          # shap_explain.py -> shap/plots/*.png
├── tests/                         # pytest: input validation + inference
├── k8s/                           # deployment.yaml, service.yaml
├── monitoring/                    # prometheus.yml + Grafana provisioning & dashboard
├── Dockerfile  docker-compose.yml
├── requirements.txt  requirements-serve.txt
└── run_pipeline.py                # ETL -> EDA -> train -> evaluate -> SHAP
```

## 1. Run the ML pipeline
```bash
pip install -r requirements.txt
python run_pipeline.py            # or run each stage separately:
python src/pipeline.py            # ETL + feature store
python eda/eda_plots.py           # EDA graphs
python models/train.py            # 7 models x {Class Weights, SMOTE}, tracked in MLflow
python evaluation/evaluate.py     # leaderboard, threshold, confusion matrix, registry
python shap/shap_explain.py       # SHAP plots
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

## 2. Streamlit app + Prometheus metrics
```bash
streamlit run src/app.py          # UI http://localhost:8501  |  metrics http://localhost:8000/metrics
```
Metrics: `ml_prediction_requests_total`, `ml_predictions_total{predicted_class}`,
`ml_prediction_latency_seconds` (histogram), `ml_prediction_errors_total{error_type}`, `ml_fraud_probability`, `ml_app_up`.

## 3. Tests
```bash
pytest
flake8 src models evaluation eda shap tests run_pipeline.py
```

## 4. Docker + monitoring stack
```bash
docker compose up --build -d
```
| Service    | URL                      |
|------------|--------------------------|
| Streamlit  | http://localhost:8501    |
| /metrics   | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090    |
| Grafana    | http://localhost:3000 (admin / admin) |

## 5. Kubernetes (Minikube)
```bash
minikube start --driver=docker
minikube image load upi-fraud-app:latest
kubectl apply -f k8s/
kubectl scale deployment upi-fraud-app --replicas=4
minikube service upi-fraud-service --url
```

## Results (hold-out test set, 560 transactions)
Best model: **CatBoost + Class Weights** – ROC-AUC 0.922, F1-Macro 0.814, Fraud recall 0.75, tuned threshold 0.51.
