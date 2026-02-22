import os
import pandas as pd
from src.utils import load_config


def main():

    # Load config
    config = load_config()

    raw_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    production_dir = config["data"]["production_dir"]
    version = config["data"]["current_version"]

    # Load raw dataset
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)

    # Basic preprocessing
    # Drop identifier columns (IMPORTANT — prevents huge feature explosion)
    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")

    # Encode ONLY the categorical feature
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Chronological split (MANDATORY)
    train_df = df.iloc[:7000]
    prod_df = df.iloc[7000:]

    # Save versioned files
    train_path = os.path.join(processed_dir, f"{version}_train.csv")
    prod_path = os.path.join(production_dir, "day1_stream.csv")

    train_df.to_csv(train_path, index=False)
    prod_df.to_csv(prod_path, index=False)

    # Update manifest.txt (manual lineage log)
    with open("manifest.txt", "a") as f:
        f.write(
            f"\n[{version}] created by data_prep.py | "
            f"raw={raw_path} | train_file={train_path}"
        )

    print("Data preparation complete.")


if __name__ == "__main__":
    main()