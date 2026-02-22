import requests
import pandas as pd
import json
from src.utils import load_config


def main():

    print("Starting production monitoring...")

    # Load config
    config = load_config()

    production_path = config["data"]["production_dir"] + "day1_stream.csv"
    threshold = config["monitoring"]["error_threshold"]
    port = config["deployment"]["port"]

    api_url = f"http://127.0.0.1:{port}/predict"

    # Load production dataset
    df = pd.read_csv(production_path)

    target_col = "Machine failure"

    if target_col not in df.columns:
        raise ValueError("Target column missing in production data")

    # Separate features and labels
    X = df.drop(columns=[target_col])
    y_true = df[target_col]

    predictions = []

    print("Sending production data to API...")

    # Call API row-by-row
    for _, row in X.iterrows():

        payload = {"features": row.tolist()}

        try:
            response = requests.post(api_url, json=payload)

            if response.status_code != 200:
                print("API error:", response.text)
                continue

            predictions.append(response.json()["prediction"])

        except Exception as e:
            print("Request failed:", e)

    # Align lengths
    predictions = predictions[:len(y_true)]

    # Calculate production error
    errors = (pd.Series(predictions) != y_true).sum()
    error_rate = errors / len(y_true)

    print(f"Production Error Rate: {error_rate}")


    # Load training baseline
    metadata_path = f"models/model_{config['data']['current_version']}_metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    training_error = 1 - metadata["accuracy"]

    print(f"Training Error Baseline: {training_error}")

    # UPDATED retraining logic
    if error_rate > (training_error + threshold):
        print("⚠️ Drift detected — RETRAIN REQUIRED")
    else:
        print("Model performance within acceptable range.")


if __name__ == "__main__":
    main()