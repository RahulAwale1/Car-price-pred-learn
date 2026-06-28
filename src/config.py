import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv('APP_NAME', "CAR PRICE ML LIFECYCLE API")
ENV = os.getenv('ENV', "developmet")

MODEL_VERSION = os.getenv('MODEL_VERSION', "v1")
MODEL_NAME = os.getenv("MODEL_NAME", "car-price-predictor")
MODEL_TYPE = os.getenv("MODEL_TYPE", "RandomForestRegressor")
INPUT_SCHEMA_VERSION = os.getenv("INPUT_SCHEMA_VERSION", "v1")
CURRENCY = os.getenv("CURRENCY", "CAD")

DATA_PATH = "data/raw/car_price.csv"
PROCESSED_DATA_PATH = "data/processed/car_price_clean.csv"
MODEL_PATH = "models/model.pkl"
BEST_MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "reports/metrics.json"
TARGET_COLUMN = "price"

TEST_SIZE = 0.2
RANDOM_STATE = 25