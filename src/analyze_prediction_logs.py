from pathlib import Path
import pandas as pd


LOG_FILE_PATH = Path("logs/prediction_logs.csv")


def main():
    if not LOG_FILE_PATH.exists():
        print("No prediction logs found.")
        return

    logs_df = pd.read_csv(LOG_FILE_PATH)

    print("Prediction Log Summary")
    print("----------------------")
    print(f"Total predictions: {len(logs_df)}")
    print(f"Average predicted price: {logs_df['predicted_price'].mean():.2f}")
    print(f"Average latency: {logs_df['latency'].mean():.4f}s")


if __name__ == "__main__":
    main()