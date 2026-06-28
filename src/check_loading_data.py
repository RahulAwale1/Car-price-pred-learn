from src.config import DATA_PATH
from src.data import load_csv, preview_data

def main():
    df = load_csv(DATA_PATH)
    preview_data(df)

if __name__ == "__main__":
    main()