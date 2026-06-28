from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

def build_linear_regression_pipeline(numerical_features, categorical_features):
    numeric_transform = Pipeline(steps=[('scaler',StandardScaler())])

    categorical_transform = Pipeline(steps=[('onehot',OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transform, numerical_features),
        ("cat", categorical_transform, categorical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model", LinearRegression())
    ])

    return model_pipeline


def build_model_pipeline(model, numerical_features, categorical_features, scale_numeric = True):

    if scale_numeric:
        numeric_transform = Pipeline(steps=[('scaler',StandardScaler())])
    else:
        numeric_transform = "passthrough"

    categorical_transform = Pipeline(steps=[('onehot',OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transform, numerical_features),
        ("cat", categorical_transform, categorical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model", model)
    ])

    return model_pipeline