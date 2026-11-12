import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("model_building")
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

def load_data(x_train_path: str, y_train_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load transformed training features and labels."""
    logger.debug("Starting training data loading")
    logger.debug("Features path: %s", x_train_path)
    logger.debug("Labels path: %s", y_train_path)

    for path in [x_train_path, y_train_path]:
        if not os.path.isfile(path):
            logger.error("Input CSV file does not exist: %s", path)
            raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    try:
        x_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path)

        logger.debug("Training features shape: %s", x_train.shape)
        logger.debug("Training labels shape: %s", y_train.shape)

        if x_train.empty:
            logger.error("Training feature dataframe is empty: %s", x_train_path)
            raise ValueError("Training feature dataframe is empty")

        if y_train.empty:
            logger.error("Training label dataframe is empty: %s", y_train_path)
            raise ValueError("Training label dataframe is empty")

        if x_train.shape[0] != y_train.shape[0]:
            logger.error(
                "Feature and label row counts do not match. Features: %s, Labels: %s",
                x_train.shape[0],
                y_train.shape[0],
            )
            raise ValueError(
                f"Feature and label row counts do not match: "
                f"{x_train.shape[0]} features vs {y_train.shape[0]} labels"
            )

        # Convert labels to a 1-dimensional Series
        if y_train.shape[1] != 1:
            logger.error("Label file must contain exactly one column. Found: %s", y_train.shape[1])
            raise ValueError("Label file must contain exactly one column")

        y_train = y_train.iloc[:, 0]
        y_train.name = "sentiment"

        logger.info("Training data loaded successfully")
        return x_train, y_train

    except Exception:
        logger.exception("Failed to load training data")
        raise


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    learning_rate: float,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    logger.debug("Starting model training")
    logger.debug("Model parameters - n_estimators: %s, learning_rate: %s", n_estimators, learning_rate)

    if n_estimators <= 0:
        logger.error("Invalid n_estimators value: %s. It must be greater than 0.", n_estimators)
        raise ValueError(f"Invalid n_estimators value: {n_estimators}. It must be greater than 0.")

    if learning_rate <= 0:
        logger.error("Invalid learning_rate value: %s. It must be greater than 0.", learning_rate)
        raise ValueError(f"Invalid learning_rate value: {learning_rate}. It must be greater than 0.")

    try:
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

        logger.debug("Fitting LightGBM classifier")
        model.fit(x_train, y_train)

        logger.info("Model training completed successfully")
        logger.debug("Trained model: %s", model)

        return model

    except Exception:
        logger.exception("Failed during model training")
        raise


def save_model(model: object, output_path: str, output_dir: str) -> None:
    """Save the trained model using pickle."""
    logger.debug("Starting model saving process")
    logger.debug("Model output directory: %s", output_dir)
    logger.debug("Model output path: %s", output_path)

    if model is None:
        logger.error("Cannot save model because the model object is None")
        raise ValueError("Cannot save model because the model object is None")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured model output directory exists: %s", output_dir)

        with open(output_path, "wb") as model_file:
            pickle.dump(model, model_file)

        logger.info("Trained model saved successfully: %s", output_path)

    except Exception:
        logger.exception("Failed to save trained model")
        raise


def main(
    x_train_path: str,
    y_train_path: str,
    n_estimators: int,
    learning_rate: float,
    output_path: str,
    output_dir: str,
) -> None:
    """Run the complete model-building pipeline."""
    logger.info("Starting model-building pipeline")

    try:
        x_train, y_train = load_data(
            x_train_path=x_train_path,
            y_train_path=y_train_path,
        )

        model = train_model(
            x_train=x_train,
            y_train=y_train,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )

        save_model(
            model=model,
            output_path=output_path,
            output_dir=output_dir,
        )

        logger.info("Model-building pipeline completed successfully")

    except Exception:
        logger.exception("Model-building pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    transformed_dir = home_dir / "data" / "transformed"
    model_dir = home_dir / "model"

    x_train_path = transformed_dir / "x_train.csv"
    y_train_path = transformed_dir / "y_train.csv"

    model_path = model_dir / "model.pkl"
    yaml_path = home_dir / "params.yaml"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("YAML configuration path: %s", yaml_path)
    logger.debug("Training features path: %s", x_train_path)
    logger.debug("Training labels path: %s", y_train_path)
    logger.debug("Model output path: %s", model_path)

    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        logger.debug("Loaded YAML configuration: %s", params)

        model_params = params["model_building"]
        n_estimators = model_params["n_estimators"]
        learning_rate = model_params["learning_rate"]

        main(
            x_train_path=str(x_train_path),
            y_train_path=str(y_train_path),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            output_path=str(model_path),
            output_dir=str(model_dir),
        )

        print("Model training completed successfully")

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        sys.exit(1)

    except KeyError as e:
        logger.exception("Missing required configuration key in params.yaml: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during model building")
        sys.exit(1)