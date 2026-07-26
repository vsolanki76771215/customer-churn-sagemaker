import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "retained"

def preprocess_data(df):
    required_columns = {
        "custid",
        "created",
        "firstorder",
        "lastorder",
        "favday",
        "city",
        TARGET,
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    df = df.copy()

    for column in ["created", "firstorder", "lastorder"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["created", "firstorder", "lastorder", TARGET])

    df["first_last_days_diff"] = (
        df["lastorder"] - df["firstorder"]
    ).dt.days

    df["created_first_days_diff"] = (
        df["firstorder"] - df["created"]
    ).dt.days

    df = df.drop(
        columns=["custid", "created", "firstorder", "lastorder"]
    )

    df = pd.get_dummies(
        df,
        columns=["favday", "city"],
        prefix=["favday", "city"],
        dtype=int
    )

    df[TARGET] = df[TARGET].astype(int)

    invalid_target = ~df[TARGET].isin([0, 1])
    if invalid_target.any():
        raise ValueError("Target retained must contain only 0 and 1.")

    return df


def save_xgboost_csv(dataframe, output_path):
    """
    SageMaker built-in XGBoost expects the label in the first column
    and normally expects CSV files without headers.
    """
    ordered_columns = [TARGET] + [
        column for column in dataframe.columns if column != TARGET
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataframe[ordered_columns].to_csv(
        output_path,
        header=False,
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/opt/ml/processing/input/storedata_total.csv"
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input_path)
    processed_df = preprocess_data(raw_df)

    train_df, temporary_df = train_test_split(
        processed_df,
        test_size=0.30,
        stratify=processed_df[TARGET],
        random_state=args.random_state
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        stratify=temporary_df[TARGET],
        random_state=args.random_state
    )

    save_xgboost_csv(
        train_df,
        "/opt/ml/processing/train/train.csv"
    )

    save_xgboost_csv(
        validation_df,
        "/opt/ml/processing/validation/validation.csv"
    )

    save_xgboost_csv(
        test_df,
        "/opt/ml/processing/test/test.csv"
    )

    os.makedirs("/opt/ml/processing/metadata", exist_ok=True)

    metadata = {
        "target": TARGET,
        "feature_names": [
            column for column in processed_df.columns
            if column != TARGET
        ],
        "row_counts": {
            "raw": len(raw_df),
            "processed": len(processed_df),
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df)
        },
        "class_distribution": (
            processed_df[TARGET]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
        )
    }

    with open(
        "/opt/ml/processing/metadata/metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()