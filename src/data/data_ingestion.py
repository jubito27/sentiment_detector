import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("data_ingestion")
    logger.setLevel(level)

    # Avoid duplicate handlers if this module is imported/run repeatedly
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(logging.DEBUG)


# ---------------- Data functions ----------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load the raw dataset and remove unnecessary columns."""
    logger.debug("Starting data load from: %s", data_path)

    if not os.path.isfile(data_path):
        logger.error("Input CSV file does not exist: %s", data_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.debug("Loaded raw dataframe with shape: %s", df.shape)

        columns_to_drop = [col for col in ["tweet_id", "Unnamed: 0"] if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
            logger.debug("Dropped columns: %s", columns_to_drop)

        if df.empty:
            logger.error("Loaded dataframe is empty: %s", data_path)
            raise ValueError("Loaded dataframe is empty")

        logger.info("Data loaded successfully. Final shape: %s", df.shape)
        return df

    except Exception:
        logger.exception("Failed to load data from: %s", data_path)
        raise


def splitting_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    logger.debug("Starting train-test split with test_size=%s", test_size)

    if df.shape[1] < 2:
        logger.error("Dataframe must contain at least two columns. Found: %s", df.shape[1])
        raise ValueError("Dataframe must contain at least two columns")

    if not 0 < test_size < 1:
        logger.error("Invalid test_size: %s. It must be between 0 and 1.", test_size)
        raise ValueError(f"Invalid test_size: {test_size}. It must be between 0 and 1.")

    try:
        # Assuming the last column is the feature and the first is the target
        x = df.iloc[:, -1]
        y = df.iloc[:, 0]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
        )

        train = pd.concat([y_train, x_train], axis=1)
        test = pd.concat([y_test, x_test], axis=1)

        logger.info("Train-test split completed. Train shape: %s, Test shape: %s", train.shape, test.shape)
        logger.debug("Train columns: %s", list(train.columns))
        logger.debug("Test columns: %s", list(test.columns))

        return train, test

    except Exception:
        logger.exception("Failed during train-test splitting")
        raise


def saving_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: str,
    test_path: str,
    output_dir: str,
) -> None:
    """Save train and test datasets as CSV files."""
    logger.debug("Starting data saving process")
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Train path: %s", train_path)
    logger.debug("Test path: %s", test_path)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured output directory exists: %s", output_dir)

        train.to_csv(train_path)
        logger.info("Train data saved successfully: %s", train_path)

        test.to_csv(test_path)
        logger.info("Test data saved successfully: %s", test_path)

    except Exception:
        logger.exception("Failed to save train/test data")
        raise


def main(
    data_path: str,
    train_path: str,
    test_path: str,
    test_size: float,
    output_dir: str,
) -> None:
    """Run the complete data ingestion pipeline."""
    logger.info("Starting data ingestion pipeline")

    try:
        df = load_data(data_path=data_path)
        train, test = splitting_data(df=df, test_size=test_size)

        saving_data(
            train=train,
            test=test,
            train_path=train_path,
            test_path=test_path,
            output_dir=output_dir,
        )

        logger.info("Data ingestion pipeline completed successfully")

    except Exception:
        logger.exception("Data ingestion pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    output_dir = home_dir / "data" / "raw"
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    yaml_path = home_dir / "params.yaml"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("YAML configuration path: %s", yaml_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Loaded YAML configuration: %s", params)

        test_size = params["data_ingestion"]["test_size"]
        data_path = "/mnt/WindowsD/Artificial Intelligence/Datasets/tweets_emotions.csv"

        main(
            data_path=str(data_path),
            train_path=str(train_path),
            test_path=str(test_path),
            test_size=test_size,
            output_dir=str(output_dir),
        )

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        sys.exit(1)

    except KeyError as e:
        logger.exception("Missing required configuration key in params.yaml: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during data ingestion")
        sys.exit(1)