import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("model_evaluation")
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


# ---------------- Model and data loading ----------------

def load_model(model_path: str) -> object:
    """Load a pickled trained model."""
    logger.debug("Starting model loading")
    logger.debug("Model path: %s", model_path)

    if not os.path.isfile(model_path):
        logger.error("Model file does not exist: %s", model_path)
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        logger.debug("Loaded model: %s", model)
        logger.info("Model loaded successfully")

        return model

    except Exception:
        logger.exception("Failed to load model from: %s", model_path)
        raise


def load_data(x_test_path: str, y_test_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load transformed test features and labels."""
    logger.debug("Starting test data loading")
    logger.debug("Test features path: %s", x_test_path)
    logger.debug("Test labels path: %s", y_test_path)

    for path in [x_test_path, y_test_path]:
        if not os.path.isfile(path):
            logger.error("Input CSV file does not exist: %s", path)
            raise FileNotFoundError(f"Input CSV file does not exist: {path}")

    try:
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)

        logger.debug("Test features shape: %s", x_test.shape)
        logger.debug("Test labels shape: %s", y_test.shape)

        if x_test.empty:
            logger.error("Test feature dataframe is empty: %s", x_test_path)
            raise ValueError("Test feature dataframe is empty")

        if y_test.empty:
            logger.error("Test label dataframe is empty: %s", y_test_path)
            raise ValueError("Test label dataframe is empty")

        if x_test.shape[0] != y_test.shape[0]:
            logger.error(
                "Feature and label row counts do not match. Features: %s, Labels: %s",
                x_test.shape[0],
                y_test.shape[0],
            )
            raise ValueError(
                f"Feature and label row counts do not match: "
                f"{x_test.shape[0]} features vs {y_test.shape[0]} labels"
            )

        if y_test.shape[1] != 1:
            logger.error("Label file must contain exactly one column. Found: %s", y_test.shape[1])
            raise ValueError("Label file must contain exactly one column")

        y_test = y_test.iloc[:, 0]
        y_test.name = "sentiment"

        logger.info("Test data loaded successfully")
        return x_test, y_test

    except Exception:
        logger.exception("Failed to load test data")
        raise


# ---------------- Evaluation ----------------

def calculate_scores(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, str]:
    """Calculate classification metrics and generate a classification report."""
    logger.debug("Starting model evaluation")

    if model is None:
        logger.error("Model object is None; cannot evaluate the model")
        raise ValueError("Model object is None; cannot evaluate the model")

    try:
        logger.debug("Generating predictions")
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        classification = classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )

        logger.debug(
            "Metrics - accuracy: %s, precision: %s, recall: %s, f1: %s",
            accuracy,
            precision,
            recall,
            f1,
        )

        y_proba = model.predict_proba(x_test)
        model_classes = np.asarray(model.classes_)
        test_classes = pd.unique(y_test)

        logger.debug("Model classes: %s", list(model_classes))
        logger.debug("Test classes: %s", list(test_classes))

        if len(test_classes) < 2:
            logger.error("ROC AUC requires at least two classes in y_test")
            raise ValueError("ROC AUC requires at least two classes in y_test")

        if not np.isin(test_classes, model_classes).all():
            logger.error(
                "y_test contains a class that the model was not trained to predict"
            )
            raise ValueError(
                "y_test contains a class that the model was not trained to predict"
            )

        class_aucs = []
        for class_label in test_classes:
            class_index = np.flatnonzero(model_classes == class_label)[0]
            y_binary = (y_test == class_label).astype(int)

            class_auc = roc_auc_score(y_binary, y_proba[:, class_index])
            class_aucs.append(class_auc)

            logger.debug("ROC AUC for class %s: %s", class_label, class_auc)

        auc = float(np.mean(class_aucs))
        logger.debug("Mean one-vs-rest ROC AUC: %s", auc)

        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

        scores_df = pd.DataFrame([scores], index=["lightgbm"]).T

        logger.info("Model evaluation completed successfully")
        return scores_df, classification

    except Exception:
        logger.exception("Failed during model evaluation")
        raise


# ---------------- Report saving ----------------

def save_reports(
    scores_df: pd.DataFrame,
    classification_report_text: str,
    scores_path: str,
    classification_path: str,
    output_dir: str,
) -> None:
    """Save evaluation metrics and classification report."""
    logger.debug("Starting report saving process")
    logger.debug("Report output directory: %s", output_dir)
    logger.debug("Scores output path: %s", scores_path)
    logger.debug("Classification report output path: %s", classification_path)

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Ensured report output directory exists: %s", output_dir)

        scores_df.to_csv(scores_path, index=True)
        logger.info("Evaluation scores saved successfully: %s", scores_path)

        Path(classification_path).write_text(classification_report_text, encoding="utf-8")
        logger.info("Classification report saved successfully: %s", classification_path)

    except Exception:
        logger.exception("Failed to save evaluation reports")
        raise


# ---------------- Pipeline ----------------

def main(
    model_path: str,
    x_test_path: str,
    y_test_path: str,
    scores_path: str,
    classification_path: str,
    output_dir: str,
) -> None:
    """Run the complete model-evaluation pipeline."""
    logger.info("Starting model-evaluation pipeline")

    try:
        model = load_model(model_path=model_path)

        x_test, y_test = load_data(
            x_test_path=x_test_path,
            y_test_path=y_test_path,
        )

        scores_df, classification_report_text = calculate_scores(
            model=model,
            x_test=x_test,
            y_test=y_test,
        )

        save_reports(
            scores_df=scores_df,
            classification_report_text=classification_report_text,
            scores_path=scores_path,
            classification_path=classification_path,
            output_dir=output_dir,
        )

        logger.info("Model-evaluation pipeline completed successfully")

    except Exception:
        logger.exception("Model-evaluation pipeline failed")
        raise


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    transformed_dir = home_dir / "data" / "transformed"
    reports_dir = home_dir / "reports" / "figures"

    model_path = home_dir / "model" / "model.pkl"

    x_test_path = transformed_dir / "x_test.csv"
    y_test_path = transformed_dir / "y_test.csv"

    scores_path = reports_dir / "scores.csv"
    classification_path = reports_dir / "classification.txt"

    logger.debug("Project home directory: %s", home_dir)
    logger.debug("Model path: %s", model_path)
    logger.debug("Test features path: %s", x_test_path)
    logger.debug("Test labels path: %s", y_test_path)
    logger.debug("Report output directory: %s", reports_dir)

    try:
        main(
            model_path=str(model_path),
            x_test_path=str(x_test_path),
            y_test_path=str(y_test_path),
            scores_path=str(scores_path),
            classification_path=str(classification_path),
            output_dir=str(reports_dir),
        )

        print("Model evaluation completed successfully")

    except FileNotFoundError:
        logger.exception("Required file or directory was not found")
        sys.exit(1)

    except ValueError as e:
        logger.exception("Validation error during model evaluation: %s", e)
        sys.exit(1)

    except Exception:
        logger.exception("Unexpected error occurred during model evaluation")
        sys.exit(1)