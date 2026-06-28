from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.pipeline import build_model_pipeline


def test_build_model_pipeline_returns_pipeline():
    pipeline = build_model_pipeline(
        model=RandomForestRegressor(),
        numerical_features=["year", "mileage"],
        categorical_features=["brand", "fuel_type"],
        scale_numeric=False
    )
    
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps