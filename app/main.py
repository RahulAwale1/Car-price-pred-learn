from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import logging
import time
import uuid
from src.schema.error_detail_schema import ErrorResponse
from src.helper.error_helper import raise_api_error
from src.config import APP_NAME, BEST_MODEL_PATH, MODEL_VERSION, MODEL_NAME, MODEL_TYPE, INPUT_SCHEMA_VERSION, CURRENCY
from typing import Literal, List
from src.prediction_logger import save_prediction_log
from src.schema.car_input_schema import CarInput
from src.schema.prediction_schema import PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from src.schema.model_info_schema import ModelInfoResponse

model_pipeline = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    logger.info(f"Loading model from {BEST_MODEL_PATH}")
    model_pipeline = joblib.load(BEST_MODEL_PATH)
    logger.info(f"Model loaded successfully.")
    yield
    logger.info("API shutting down.")

app = FastAPI(
    title= APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return{
        "status": "Ok",
        "model_loaded": model_pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse, responses={
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
})
def predict(input_data: CarInput):
    request_id = str(uuid.uuid4())[:8]
    if model_pipeline is None:
        logger.info("Prediction requested but model is not loaded")
        raise_api_error(
            status_code=503,
            error_code="MODEL_NOT_LOADED",
            message="Model is not loaded",
            request_id=request_id
        )
    
    start_time = time.time()
    try:
        logger.info(f"Received prediction request: {input_data.model_dump()}")
        input_df = pd.DataFrame([input_data.model_dump()])
        prediction = model_pipeline.predict(input_df)

        latency = round(time.time() - start_time, 4)
        save_prediction_log(
            request_id=request_id,
            input_data=input_data.model_dump(),
            predicted_price=round(float(prediction[0]), 2),
            model_version=MODEL_VERSION,
            latency=latency
        )
        logger.info(f"Prediction successful: request_id = {request_id} | price = {float(prediction[0])} | latency = {latency} | model_version = {MODEL_VERSION}")
        return PredictionResponse(predicted_price=float(prediction[0]), currency="CAD", index=0)
    
    except Exception as e:
        logger.info(f"Prediction failed. | request_id = {request_id}")
        raise_api_error(
            status_code=500,
            error_code="PREDICTION_FAILED",
            message="Prediction failed. Contact support with request_id.",
            request_id=request_id
        )
    

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(batch_data: BatchPredictionRequest):
    request_id = str(uuid.uuid4())[:8]

    if model_pipeline is None:
        logger.error(f"request_id: {request_id}, batch prediction requested but model is not loaded.")
        raise_api_error(
            status_code=503,
            error_code="MODEL_NOT_LOADED",
            message="Model is not loaded",
            request_id=request_id
        )

    start_time = time.time()
    try:
        logger.info(f"Received Batch Prediction request: request_id = {request_id}, count = {len(batch_data.items)}")
        input_records = [item.model_dump() for item in batch_data.items]

        input_df = pd.DataFrame(input_records)
        raw_predictions = model_pipeline.predict(input_df)
        print(raw_predictions)

        predictions = [
            PredictionResponse(
                index=index,
                predicted_price=round(float(prediction), 2),
                currency="CAD"
            )
            for index,prediction in enumerate(raw_predictions)
        ]

        latency = round(time.time() - start_time, 4)
        logger.info(
            f"request_id={request_id} | "
            f"Batch prediction successful | "
            f"count={len(predictions)} | "
            f"model_version={MODEL_VERSION} | "
            f"latency={latency}s"
        )
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_version=MODEL_VERSION,
            request_id=request_id
        )
    except Exception as e:
        logger.exception(f"request_id={request_id} | Batch prediction failed")
        raise_api_error(
            status_code=500,
            error_code="BATCH_PREDICTION_FAILED",
            message="Batch Prediction Failed. Contact support with request_id.",
            request_id=request_id
        )
    
@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        input_schema_version=INPUT_SCHEMA_VERSION,
        currency=CURRENCY,
        model_loaded=model_pipeline is not None
    )

