from src.config import MODEL_PATH
import pandas as pd
import joblib

def load_model(path: str = MODEL_PATH):
    return joblib.load(path)

def predict_price(input_data: dict) -> float:

    model_pipeline = load_model()
    input_df = pd.DataFrame([input_data])
    prediction = model_pipeline.predict(input_df)

    return float(prediction[0])

def main():
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

    predicted_price = predict_price(sample_car)
    print("Sample input:")
    print(sample_car)
    print(f"\nPredicted price: {predicted_price:.2f}")

if __name__ == "__main__":
    main()


