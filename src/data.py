from pathlib import Path
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found:{path}")
    
    return pd.read_csv(file_path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(output_path.exists(), output_path.is_dir(), output_path.is_file())
    df.to_csv(output_path, index=False)


def preview_data(df: pd.DataFrame, rows: int = 5) -> None:
    
    print("Shape:")
    print(df.shape)
    print("\n First 5 rows:")
    print(df.head(rows))
    print("Column types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isnull().sum())
