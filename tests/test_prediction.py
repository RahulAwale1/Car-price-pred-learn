
import pandas as pd

from src.config import BEST_MODEL_PATH, MODEL_PATH
from src.predict import load_model


def test_prediction_result_numeric():
    model_pipeline = load_model(path=BEST_MODEL_PATH)
    
    sample_car = {
        # Numerical features
        "symboling": 3,
        "wheelbase": 88.6,
        "carlength": 168.8,
        "carwidth": 64.1,
        "carheight": 48.8,
        "curbweight": 2548,
        "enginesize": 130,
        "boreratio": 3.47,
        "stroke": 2.68,
        "compressionratio": 9.0,
        "horsepower": 111,
        "peakrpm": 5000,
        "citympg": 7,
        "highwaympg": 10,

        # Categorical features
        "fueltype": "gas",
        "aspiration": "std",
        "doornumber": "two",
        "carbody": "convertible",
        "drivewheel": "rwd",
        "enginelocation": "front",
        "enginetype": "dohc",
        "cylindernumber": "four",
        "fuelsystem": "mpfi",
        "brand": "alfa-romero"
    }
    df = pd.DataFrame([sample_car])
    
    prediction = model_pipeline.predict(df)
    predicted_price = prediction[0]
    
    assert isinstance(predicted_price,(int, float))
    
    