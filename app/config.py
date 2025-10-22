import os
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", r"D:\Data science\Sentiment_Analysis")
DATA_PATH = os.getenv("DATA_PATH", os.path.join(DATA_DIR, "Telco_customer_churn_status.xlsx"))
RESULTS_PATH = os.getenv("RESULTS_PATH", os.path.join(DATA_DIR, "Telco_customer_Sentiment_results.xlsx"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/ChaitanyaC20/Sentiment_Analysis.mlflow/")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "Sentiment")

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "runs")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "fine_tuned_telco_sentiment")
MODEL_FILE = os.path.join(ARTIFACT_DIR, "model.pkl")
FEATURES_FILE = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
METRICS_FILE = os.path.join(ARTIFACT_DIR, "metrics.json")

os.makedirs(ARTIFACT_DIR, exist_ok=True)