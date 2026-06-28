from src.config import PROCESSED_DATA_PATH,RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, MODEL_PATH
from src.pipeline import build_linear_regression_pipeline
from src.data import load_csv
from src.features import split_features_target, split_train_test

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib

def main():
    df = load_csv(PROCESSED_DATA_PATH)

    X,y = split_features_target(df=df, target_column=TARGET_COLUMN)

    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


    model_pipeline = build_linear_regression_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Pipeline Results")
    print("----------------------------------")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline,model_path)

    print(f"\nSaved trained pipeline to: {model_path}")

    

if __name__ == "__main__":

    main()