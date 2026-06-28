from src.data import load_csv
from src.config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE
from src.features import split_features_target, split_train_test
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

def main():
    df = load_csv(PROCESSED_DATA_PATH)
    X,y = split_features_target(df=df, target_column='price')
    X_train, X_test, y_train, y_test = split_train_test(X,y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    baseline_model = DummyRegressor(strategy='mean')

    baseline_model.fit(X_train, y_train)

    y_pred = baseline_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print("Baseline Model Results")
    print("-------------------------")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

if __name__ == '__main__':
    main()