import pandas as pd
import pytest

from src.features import split_features_target

def test_split_features_target_seperates_x_and_y():
    df = pd.DataFrame({
            "brand": ["Toyota", "Honda"],
            "year": [2020, 2021],
            "price": [25000, 27000]
        })
    X, y = split_features_target(df, "price")
    
    assert "price" not in X.columns
    assert y.name == "price"
    assert len(X) == 2
    assert len(y) == 2

def test_split_features_target_raises_error_when_target_missing():
    df = pd.DataFrame({
        "brand": ["Toyota"],
        "year": [2020]
    })

    with pytest.raises(ValueError):
        split_features_target(df, "price")