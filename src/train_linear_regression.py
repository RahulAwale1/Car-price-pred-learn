from src.data import load_csv
from src.features import (split_features_target, split_train_test, encode_categorical_features)
from src.config import PROCESSED_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    df =load_csv(PROCESSED_DATA_PATH)
    X,y = split_features_target(df=df, target_column=TARGET_COLUMN)
    X_encoded = encode_categorical_features(X)
    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression Results")
    print("-------------------------")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # print("\nFeature Coefficients")
    # print("--------------------")
    # for feature, coefficient in zip(X_encoded.columns, model.coef_):
    #     print(f"{feature}: {coefficient:.2f}")
    # print(f"\nIntercept: {model.intercept_:.2f}")
    

if __name__ == "__main__":
    main()