from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "loan_status"
RANDOM_STATE = 42

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
SCALE_COLS = NUMERIC_COLS + ["loan_grade"]
NOMINAL_COLS = ["person_home_ownership", "loan_intent"]
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
DEFAULT_MAP = {"N": 0, "Y": 1}


def load_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outlier_mask = (df["person_age"] > 100) | (df["person_emp_length"] > 60)
    return df.loc[~outlier_mask].copy()


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df.copy(), test_df.copy()


def handle_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    medians = train_df[NUMERIC_COLS].median(numeric_only=True)

    for col in NUMERIC_COLS:
        train_df[col] = train_df[col].fillna(medians[col])
        test_df[col] = test_df[col].fillna(medians[col])

    return train_df, test_df, medians


def winsorize_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    q01: pd.Series | None = None,
    q99: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if q01 is None:
        q01 = train_df[NUMERIC_COLS].quantile(0.01)
    if q99 is None:
        q99 = train_df[NUMERIC_COLS].quantile(0.99)

    for col in NUMERIC_COLS:
        train_df[col] = train_df[col].clip(q01[col], q99[col])
        test_df[col] = test_df[col].clip(q01[col], q99[col])

    return train_df, test_df, q01, q99


def encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in (train_df, test_df):
        df["loan_grade"] = df["loan_grade"].map(GRADE_MAP).astype(int)
        df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(DEFAULT_MAP).astype(int)

    train_df = pd.get_dummies(train_df, columns=NOMINAL_COLS, drop_first=False, dtype=int)
    test_df = pd.get_dummies(test_df, columns=NOMINAL_COLS, drop_first=False, dtype=int)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
    return train_df, test_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df[SCALE_COLS] = scaler.fit_transform(train_df[SCALE_COLS])
    test_df[SCALE_COLS] = scaler.transform(test_df[SCALE_COLS])
    return train_df, test_df, scaler


def save_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_columns = [col for col in train_df.columns if col != TARGET] + [TARGET]
    train_df = train_df[ordered_columns]
    test_df = test_df[ordered_columns]
    processed_df = pd.concat([train_df, test_df], axis=0)

    train_df.to_csv(output_dir / "credit_risk_train.csv", index=False)
    test_df.to_csv(output_dir / "credit_risk_test.csv", index=False)
    processed_df.to_csv(output_dir / "credit_risk_processed.csv", index=False)
    return processed_df


def run_preprocessing(input_path: Path, output_dir: Path) -> dict[str, object]:
    raw_df = load_data(input_path)
    missing_before = raw_df.isna().sum().to_dict()
    duplicate_before = int(raw_df.duplicated().sum())

    deduplicated_df = remove_duplicates(raw_df)
    clean_df = handle_outliers(deduplicated_df)
    removed_outliers = int(len(deduplicated_df) - len(clean_df))

    train_df, test_df = split_dataset(clean_df)
    train_df, test_df, q01, q99 = winsorize_numeric(train_df, test_df)
    train_df, test_df, medians = handle_missing_values(train_df, test_df)
    train_df, test_df = encode_features(train_df, test_df)
    train_df, test_df, _ = scale_features(train_df, test_df)
    processed_df = save_dataset(train_df, test_df, output_dir)

    metadata = {
        "source_dataset": str(input_path),
        "target_column": TARGET,
        "initial_shape": list(raw_df.shape),
        "missing_before": missing_before,
        "duplicate_rows_removed": duplicate_before,
        "business_outlier_rows_removed": removed_outliers,
        "final_clean_shape_before_split": list(clean_df.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "processed_shape": list(processed_df.shape),
        "target_distribution_train": train_df[TARGET].value_counts().sort_index().to_dict(),
        "target_distribution_test": test_df[TARGET].value_counts().sort_index().to_dict(),
        "numeric_columns_scaled": SCALE_COLS,
        "numeric_imputation": "median fitted on train set",
        "winsorization": "1st/99th percentile fitted on train set",
        "encoding": {
            "loan_grade": "ordinal A=1..G=7",
            "cb_person_default_on_file": "binary N=0, Y=1",
            "person_home_ownership": "one-hot",
            "loan_intent": "one-hot",
        },
        "random_state": RANDOM_STATE,
    }

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated preprocessing for Credit Risk dataset.")
    parser.add_argument(
        "--input",
        default="../dataset_raw/credit_risk_dataset.csv.zip",
        help="Path to raw Credit Risk dataset zip/csv.",
    )
    parser.add_argument(
        "--output",
        default="dataset_preprocessing",
        help="Directory to store preprocessed datasets.",
    )
    args = parser.parse_args()

    metadata = run_preprocessing(Path(args.input), Path(args.output))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
