import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("feature_engineering")
    logger.setLevel(level)

    # Prevent duplicate handlers when the module is imported or re-run
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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test datasets and remove missing values."""
    logger.debug("Starting data loading")
    logger.debug("Train path: %s", train_path)
    logger.debug("Test path: %s", test_path)

    for path in [train_path, test_path]:
        if not os.path.isfile(path):
            logger.error("Input CSV file does not exist: %s", path)
            raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        logger.debug("Raw train shape: %s", train.shape)
        logger.debug("Raw test shape: %s", test.shape)

        train = train.dropna()
        test = test.dropna()

        logger.debug("Train shape after dropping missing values: %s", train.shape)
        logger.debug("Test shape after dropping missing values: %s", test.shape)

        required_columns = ["content", "sentiment"]
        for name, df in [("train", train), ("test", test)]:
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                logger.error(
                    "%s data is missing required columns: %s",
                    name,
                    missing_columns,
                )
                raise KeyError(f"{name} data is missing required columns: {missing_columns}")

        if train.empty or test.empty:
            logger.error("Train or test dataframe is empty after dropping missing values")
            raise ValueError("Train or test dataframe is empty after dropping missing values")

        logger.info("Train and test data loaded successfully")
        return train, test

    except Exception:
        logger.exception("Failed to load train/test data")
        raise


def encoding_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create TF-IDF features and encode sentiment labels."""
    logger.debug("Starting feature encoding")
    logger.debug("TF-IDF max_features: %s", max_features)

    if max_features <= 0:
        logger.error("Invalid max_features value: %s. It must be greater than 0.", max_features)
        raise ValueError(f"Invalid max_features value: {max_features}. It must be greater than 0.")

    try:
        cv = CountVectorizer(max_features=max_features)

        logger.debug("Fitting TF-IDF vectorizer on training content")
        x_train_sparse = cv.fit_transform(train["content"])

        logger.debug("Transforming test content using fitted TF-IDF vectorizer")
        x_test_sparse = cv.transform(test["content"])

        logger.debug("TF-IDF feature matrix shapes - train: %s, test: %s", x_train_sparse.shape, x_test_sparse.shape)

        x_train = pd.DataFrame(
            x_train_sparse.toarray(),
            columns=cv.get_feature_names_out(),
        )

        x_test = pd.DataFrame(
            x_test_sparse.toarray(),
            columns=cv.get_feature_names_out(),
        )

        label_encoder = LabelEncoder()

        logger.debug("Fitting label encoder on training sentiment labels")
        y_train = pd.Series(
            label_encoder.fit_transform(train["sentiment"]),
            name="sentiment",
        )

        logger.debug("Transforming test sentiment labels using fitted label encoder")
        y_test = pd.Series(
            label_encoder.transform(test["sentiment"]),
            name="sentiment",
        )

        logger.debug("Encoded label classes: %s", list(label_encoder.classes_))
        logger.info("Feature encoding completed successfully")

        return x_train, x_test, y_train, y_test

    except Exception:
        logger.exception("Failed during feature encoding")
        raise


def save_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
    output_dir: str,
) -> None:
    """Save transformed features and encoded labels as CSV files."""
    logger.debug("Starting transformed data saving process")
    logger.debug("Output directory: %s", output_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured output directory exists: %s", output_dir)

        x_train.to_csv(x_train_path, index=False)
        logger.info("Training features saved successfully: %s", x_train_path)

        x_test.to_csv(x_test_path, index=False)
        logger.info("Testing features saved successfully: %s", x_test_path)

        y_train.to_csv(y_train_path, index=False)
        logger.info("Training labels saved successfully: %s", y_train_path)

        y_test.to_csv(y_test_path, index=False)
        logger.info("Testing labels saved successfully: %s", y_test_path)

    except Exception:
        logger.exception("Failed to save transformed train/test data")
        raise


def main(
    train_path: str,
    test_path: str,
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
    output_dir: str,
    max_features: int,
) -> None:
    """Run the complete feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")

    try:
        train, test = load_data(
            train_path=train_path,
            test_path=test_path,
        )

        x_train, x_test, y_train, y_test = encoding_columns(
            train=train,
            test=test,
            max_features=max_features,
        )

        save_data(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            output_dir=output_dir,
        )

        logger.info("Feature engineering pipeline completed successfully")

    except Exception:
        logger.exception("Feature engineering pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    processed_dir = home_dir / "data" / "processed"
    transformed_dir = home_dir / "data" / "transformed"

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    x_train_path = transformed_dir / "x_train.csv"
    x_test_path = transformed_dir / "x_test.csv"
    y_train_path = transformed_dir / "y_train.csv"
    y_test_path = transformed_dir / "y_test.csv"

    yaml_path = home_dir / "params.yaml"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("YAML configuration path: %s", yaml_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Loaded YAML configuration: %s", params)

        max_features = params["feature_engineering"]["max_features"]

        main(
            train_path=str(train_path),
            test_path=str(test_path),
            x_train_path=str(x_train_path),
            x_test_path=str(x_test_path),
            y_train_path=str(y_train_path),
            y_test_path=str(y_test_path),
            output_dir=str(transformed_dir),
            max_features=max_features,
        )

        print("Done")

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        sys.exit(1)

    except KeyError as e:
        logger.exception("Missing required configuration key in params.yaml: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during feature engineering")
        sys.exit(1)