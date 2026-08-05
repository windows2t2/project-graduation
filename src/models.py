"""
src/models.py
Machine Learning: salary prediction (regression) & job-category classification.
"""

import os
import pickle
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

from src.utils import MODELS_DIR, logger

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Salary Prediction (Regression)
# ======================================================================

REGRESSION_MODELS = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(random_state=42),
    "xgboost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
    "lightgbm": lgb.LGBMRegressor(random_state=42, verbose=-1),
}

REGRESSION_PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.1],
    },
}


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dictionary of regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune: bool = False,
) -> dict:
    """
    Train and evaluate all regression models.
    Returns a dict: {model_name: {"model": ..., "metrics": {...}, "predictions": [...]}}
    """
    results = {}

    for name, model in REGRESSION_MODELS.items():
        logger.info("Training regression model: %s", name)

        if tune and name in REGRESSION_PARAM_GRIDS:
            grid = GridSearchCV(
                model, REGRESSION_PARAM_GRIDS[name],
                cv=3, scoring="neg_mean_squared_error", n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info("Best params for %s: %s", name, grid.best_params_)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred,
        }
        logger.info("%s → MAE=%.2f  RMSE=%.2f  R²=%.3f", name, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    return results


def save_model(model: Any, filename: str) -> str:
    """Pickle a trained model to the models/ directory."""
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", path)
    return str(path)


def load_model(filename: str) -> Any:
    """Load a pickled model."""
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
# Job Category Classification
# ======================================================================

CLASSIFICATION_MODELS = {
    "logistic": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
    "xgboost": xgb.XGBClassifier(objective="multi:softmax", random_state=42),
    "lightgbm": lgb.LGBMClassifier(random_state=42, verbose=-1),
}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Train and evaluate classification models."""
    results = {}

    for name, model in CLASSIFICATION_MODELS.items():
        logger.info("Training classification model: %s", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_classification(y_test, y_pred)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "predictions": y_pred,
        }
        logger.info("%s → Accuracy=%.3f  F1=%.3f", name, metrics["accuracy"], metrics["f1_weighted"])

    return results


def select_best_model(results: dict, metric: str = "R2", higher_is_better: bool = True) -> Tuple[str, Any]:
    """Return (name, model) of the best model by the given metric."""
    best_name = None
    best_score = -float("inf") if higher_is_better else float("inf")

    for name, info in results.items():
        score = info["metrics"].get(metric, 0)
        if (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
            best_score = score
            best_name = name

    logger.info("Best model: %s (%s = %.4f)", best_name, metric, best_score)
    return best_name, results[best_name]["model"]
