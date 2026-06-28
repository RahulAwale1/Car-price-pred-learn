import pandas as pd
import pytest

from src.data import load_csv, save_csv


def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({
        "brand": ["Toyota"],
        "year": [2020],
        "price": [25000]
    })

    file_path = tmp_path / "test_data.csv"

    save_csv(df, str(file_path))
    loaded_df = load_csv(str(file_path))

    assert loaded_df.shape == df.shape
    assert loaded_df.columns.tolist() == df.columns.tolist()
    assert loaded_df.loc[0, "brand"] == "Toyota"


def test_load_csv_raises_error_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv("missing_file.csv")