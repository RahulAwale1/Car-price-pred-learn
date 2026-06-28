from pathlib import Path
from datetime import datetime, timezone
import csv

LOG_FILE_PATH = Path("logs/prediction_logs.csv")

FIELDNAMES = [
    "timestamp",
    "request_id",
    "input_data",
    "predicted_price",
    "model_version",
    "latency"
]

def save_prediction_log(
        request_id: str,
        input_data: dict,
        predicted_price: float,
        model_version: str,
        latency: float
) -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True,
                                exist_ok=True)
    
    file_exits = LOG_FILE_PATH.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "input_data": input_data,
        "predicted_price": predicted_price,
        "model_version": model_version,
        "latency": latency

    }

    with open(LOG_FILE_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exits:
            writer.writeheader()
        
        writer.writerow(row)
