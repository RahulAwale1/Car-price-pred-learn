from src.data import load_csv
from src.features import (split_features_target, split_train_test, encode_categorical_features)
from src.config import PROCESSED_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, BEST_MODEL_PATH
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.pipeline import build_model_pipeline
from pathlib import Path
import joblib
import mlflow, mlflow.sklearn
import json
from sklearn.model_selection import cross_validate, KFold

def evaluate_model(model_pipeline, X_test, y_test, prefix: str):

    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        f"{prefix}_mae": mae,
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2
    }


def main():
    mlflow.set_experiment("car-price-model-comparision")

    df = load_csv(PROCESSED_DATA_PATH)

    X, y = split_features_target(df=df, target_column=TARGET_COLUMN)

    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


    X_train, X_test, y_train, y_test = split_train_test(X=X, y=y, random_state=RANDOM_STATE, test_size=TEST_SIZE)

    models = {
        "dummy_mean": {
            "model": DummyRegressor(strategy="mean"),
            "scale_numeric": False,
        },
        "Linear_regression": {
            "model": LinearRegression(),
            "scale_numeric": True,

        },
        "random_forest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, max_depth=5),
            "scale_numeric": False,
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=RANDOM_STATE, max_depth=3),
            "scale_numeric": False,
        }
    }

    results = {}

    best_model = None
    best_model_pipeline = None
    best_rmse = float('inf')

    for model_name, config in models.items():
        print(f"training model: {model_name}")
        
        model_pipeline = build_model_pipeline(
            model=config['model'],
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scale_numeric=config['scale_numeric']
        )

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("numeric_features", ", " .join(numerical_features))
            mlflow.log_param("categorical_features", ", ".join(categorical_features))
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("scale_numeric", config['scale_numeric'])

            model_pipeline.fit(X=X_train, y=y_train)

            train_metrics = evaluate_model(model_pipeline=model_pipeline, X_test=X_train, y_test= y_train, prefix="train")
            test_metrics = evaluate_model(model_pipeline=model_pipeline, X_test=X_test, y_test= y_test,prefix="test")
            
            metrics = {**train_metrics, **test_metrics}
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="compare_models"
            )

            results[model_name] = metrics

            print(f"train rmse: {metrics['train_rmse']:.2f}")
            print(f"test rmse: {metrics['test_rmse']:.2f}")
            print(f"train r2: {metrics['train_r2']:.2f}")
            print(f"test r2: {metrics['test_r2']:.2f}")

            if metrics['test_rmse'] < best_rmse:
                best_rmse = metrics["test_rmse"]
                best_model = model_name
                best_model_pipeline = model_pipeline
    
    model_path = Path(BEST_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_pipeline, model_path)

    reports_path = Path("reports")
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    results_file = reports_path / "model_comparision.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print("------------Model Comparison----------------")
    print("\n")

    for model_name, metric_value in results.items():
        print("model:", model_name)
        print(f"train rmse: {metric_value['train_rmse']:.2f}")
        print(f"test rmse: {metric_value['test_rmse']:.2f}")
        print(f"train r2: {metric_value['train_r2']:.2f}")
        print(f"test r2: {metric_value['test_r2']:.2f}")

    print("\nBest Model:", best_model)
    print(f"Best Test RMSE: {best_rmse:.2f}")
    print(f"Saved best model to: {model_path}")
    print(f"Saved comparison report to: {results_file}")



if __name__ == "__main__":
    main()