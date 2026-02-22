import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import load_config


def get_git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "no_git_repo"


def main():

    print("Starting training pipeline...")

    # Load config.yaml
    config = load_config()

    processed_dir = config["data"]["processed_dir"]
    version = config["data"]["current_version"]
    model_path = config["deployment"]["model_path"]

    data_path = os.path.join(processed_dir, f"{version}_train.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)

    # IMPORTANT: dataset column has SPACE not underscore
    target_col = "Machine failure"

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Dataset loaded: {X.shape}")

    # Model parameters from config.yaml
    params = config["model_params"]

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )

    # Train model
    model.fit(X, y)

    # Evaluate (training accuracy only — assignment allows this)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    print(f"Training Accuracy: {acc}")

    # Save model artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model saved at {model_path}")

    # Create metadata JSON
    metadata = {
        "model_version": version,
        "dataset_version": version,
        "training_date": str(datetime.now()),
        "accuracy": acc,
        "git_commit": get_git_commit(),
        "model_params": params
    }

    metadata_path = f"models/model_{version}_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved at {metadata_path}")

    # Update manual Model Registry log
    log_path = "models/model_metadata.log"

    with open(log_path, "a") as f:
        f.write(
            f"\nmodel_{version} | data={version} | acc={acc} | commit={metadata['git_commit']}"
        )

    print("Model registry updated.")
    print("Training pipeline complete.")


if __name__ == "__main__":
    main()