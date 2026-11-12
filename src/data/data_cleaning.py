import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

import automac


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("data_processing")
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

def clean_features(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train/test CSVs and transform the text feature."""
    logger.debug("Starting feature cleaning")
    logger.debug("Train input path: %s", train_path)
    logger.debug("Test input path: %s", test_path)

    for path in [train_path, test_path]:
        if not os.path.isfile(path):
            logger.error("Input CSV file does not exist: %s", path)
            raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    try:
        train = pd.read_csv(train_path, index_col=0)
        test = pd.read_csv(test_path, index_col=0)

        logger.debug("Train data loaded with shape: %s", train.shape)
        logger.debug("Test data loaded with shape: %s", test.shape)
        logger.debug('Feature names before transformation: %s' , train.columns)

        if "content" not in train.columns:
            logger.error("Train data does not contain the required 'content' column")
            raise KeyError("Train data does not contain the required 'content' column")

        if "content" not in test.columns:
            logger.error("Test data does not contain the required 'content' column")
            raise KeyError("Test data does not contain the required 'content' column")

        logger.debug("Transforming 'content' column in train data")
        train["content"] = train["content"].apply(automac.text.transform)

        logger.debug("Transforming 'content' column in test data")
        test["content"] = test["content"].apply(automac.text.transform)
        logger.debug('Feature names after transformation: %s' , train.columns)
        logger.info("Feature cleaning completed successfully")
        return train, test

    except Exception:
        logger.exception("Failed during feature cleaning")
        raise


def saving_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_path: str,
    test_path: str,
    output_dir: str,
) -> None:
    """Save processed train and test datasets as CSV files."""
    logger.debug("Starting data saving process")
    logger.debug("Output directory: %s", output_dir)
    logger.debug("Train output path: %s", train_path)
    logger.debug("Test output path: %s", test_path)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured output directory exists: %s", output_dir)

        train.to_csv(train_path, index=False)
        logger.info("Processed train data saved successfully: %s", train_path)

        test.to_csv(test_path, index=False)
        logger.info("Processed test data saved successfully: %s", test_path)

    except Exception:
        logger.exception("Failed to save processed train/test data")
        raise


def main(
    train_path: str,
    test_path: str,
    output_dir: str,
    processed_train_path: str,
    processed_test_path: str,
) -> None:
    """Run the complete data processing pipeline."""
    logger.info("Starting data processing pipeline")

    try:
        train, test = clean_features(
            train_path=train_path,
            test_path=test_path,
        )

        saving_data(
            train=train,
            test=test,
            train_path=processed_train_path,
            test_path=processed_test_path,
            output_dir=output_dir,
        )

        logger.info("Data processing pipeline completed successfully")

    except Exception:
        logger.exception("Data processing pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    raw_dir = home_dir / "data" / "raw"
    processed_dir = home_dir / "data" / "processed"

    raw_train_path = raw_dir / "train.csv"
    raw_test_path = raw_dir / "test.csv"

    processed_train_path = processed_dir / "train.csv"
    processed_test_path = processed_dir / "test.csv"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("Raw train path: %s", raw_train_path)
    logger.debug("Raw test path: %s", raw_test_path)
    logger.debug("Processed output directory: %s", processed_dir)

    try:
        main(
            train_path=str(raw_train_path),
            test_path=str(raw_test_path),
            output_dir=str(processed_dir),
            processed_train_path=str(processed_train_path),
            processed_test_path=str(processed_test_path),
        )

        print("Process Done---")

    except FileNotFoundError:
        logger.exception("Required input file was not found")
        sys.exit(1)

    except KeyError as e:
        logger.exception("Required column is missing: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during data processing")
        sys.exit(1)