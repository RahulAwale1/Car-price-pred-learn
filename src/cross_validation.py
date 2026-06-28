import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate, KFold

from src.config import DATA_PATH, TARGET_COLUMN, RANDOM_STATE
from src.data import load_csv
from src.features import split_features_target
from src.pipeline import build_model_pipeline


def main():
    df = load_csv(DATA_PATH)

    X, y = split_features_target(df, TARGET_COLUMN)

    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = {
        "dummy_mean": {
            "model": DummyRegressor(strategy="mean"),
            "scale_numeric": False
        },
        "linear_regression": {
            "model": LinearRegression(),
            "scale_numeric": True
        },
        "random_forest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE
            ),
            "scale_numeric": False
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(
                max_depth=3,
                random_state=RANDOM_STATE
            ),
            "scale_numeric": False
        }
    }

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    print("Cross-Validation Results")
    print("------------------------")

    for model_name, config in models.items():
        model_pipeline = build_model_pipeline(
            model=config["model"],
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scale_numeric=config["scale_numeric"]
        )

        cv_results = cross_validate(
            model_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )

        train_rmse = -cv_results["train_rmse"]
        test_rmse = -cv_results["test_rmse"]

        train_mae = -cv_results["train_mae"]
        test_mae = -cv_results["test_mae"]

        train_r2 = cv_results["train_r2"]
        test_r2 = cv_results["test_r2"]

        print(f"\nModel: {model_name}")
        print(f"Train RMSE: {np.mean(train_rmse):.2f} ± {np.std(train_rmse):.2f}")
        print(f"Test RMSE:  {np.mean(test_rmse):.2f} ± {np.std(test_rmse):.2f}")
        print(f"Train MAE:  {np.mean(train_mae):.2f} ± {np.std(train_mae):.2f}")
        print(f"Test MAE:   {np.mean(test_mae):.2f} ± {np.std(test_mae):.2f}")
        print(f"Train R2:   {np.mean(train_r2):.2f} ± {np.std(train_r2):.2f}")
        print(f"Test R2:    {np.mean(test_r2):.2f} ± {np.std(test_r2):.2f}")


if __name__ == "__main__":
    main()