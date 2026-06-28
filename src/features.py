import pandas as pd
from src.config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE
from src.data import load_csv
from sklearn.model_selection import train_test_split


def split_features_target(df: pd.DataFrame, target_column: str):

    if not target_column in df.columns:
        raise ValueError(f"Target Column {target_column} not found in DataFrame.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X,y

def split_train_test(X, y, test_size: float = 0.2, random_state: int = 25):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def encode_categorical_features(X: pd.DataFrame):
    return pd.get_dummies(data=X, drop_first=True).astype(int)

def main():
    df = load_csv(PROCESSED_DATA_PATH)
    X,y = split_features_target(df=df, target_column="price")
    X_train, X_test, y_train, y_test = split_train_test(X,y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # print("Features Columns:")
    # print(X.columns.to_list())

    # print("\nTarget Value:")
    # print(y.name)

    # print("\nX Shape: ", X.shape)
    # print("y Shape: ", y.shape)

    # print("First rows of features:")
    # print(X.head())

    # print("First rows of target:")
    # print(y.head())

    print("Full dataset shape:", X.shape)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nTraining features:")
    print(X_train)
    print("\nTesting features:")
    print(X_test)

if __name__ == "__main__":
    main()

    