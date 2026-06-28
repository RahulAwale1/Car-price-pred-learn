from pathlib import Path
import pandas as pd

LOG_FILE_PATH = Path("logs/prediction_logs.csv")

def load_prediction_logs() -> pd.DataFrame:
    if not LOG_FILE_PATH.exists():
        raise FileNotFoundError("No prediction logs found.")
    return pd.read_csv(LOG_FILE_PATH)

def print_basic_metrics(logs_df: pd.DataFrame)-> None:
    print("Basic Monitoring Dasboard")
    print("===========================")

    print(f"Total prediction rows: {len(logs_df)}")

    print(f"\nAverage Latency: {logs_df["latency"].mean():.4f}s")
    print(f"Max Latency: {logs_df['latency'].max():.4f}s")
    print(f"P50 latency: {logs_df['latency'].quantile(0.50):.4f}s")
    print(f"P95 latency: {logs_df['latency'].quantile(0.95):.4f}s")
    print(f"P99 latency: {logs_df['latency'].quantile(0.99):.4f}s")

    print(f"\nAverage predicted price: {logs_df["predicted_price"].mean():.4f}")
    print(f"Max predicted price: {logs_df["predicted_price"].max():.4f}")
    print(f"Min predicted price: {logs_df["predicted_price"].min():.4f}")


def main():
    try:
        logs_df = load_prediction_logs()
    except FileNotFoundError as e:
        print(e)
        return
    print_basic_metrics(logs_df=logs_df)

if __name__ == "__main__":
    main()


