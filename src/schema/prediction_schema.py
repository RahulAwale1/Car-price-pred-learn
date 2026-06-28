from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List
from src.schema.car_input_schema import CarInput

class PredictionResponse(BaseModel):
    index: int
    predicted_price: float
    currency: str

class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[CarInput] = Field(..., min_length=1, max_length=100, description="List of cars to predict.")

class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: List[PredictionResponse]
    count: int
    request_id: str
    model_version: str
