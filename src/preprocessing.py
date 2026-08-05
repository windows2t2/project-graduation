"""
src/preprocessing.py
Data cleaning, feature engineering, and train/test splitting.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils import logger


# ---------------------------------------------------------------------------
# Column name mappings (normalise different dataset schemas)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # Data Science Salaries dataset columns → canonical names
    "work_year": "work_year",
    "experience_level": "experience_level",
    "employment_type": "employment_type",
    "job_title": "job_title",
    "salary": "salary",
    "salary_currency": "salary_currency",
    "salary_in_usd": "salary_in_usd",
    "employee_residence": "employee_residence",
    "remote_ratio": "remote_ratio",
    "company_location": "company_location",
    "company_size": "company_size",
}


def clean_salaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Data Science Salaries dataset.
    - Drop duplicates & rows with missing salary
    - Convert salary to USD where needed
    - Standardise column names if they vary
    """
    df = df.copy()

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows without salary
    if "salary_in_usd" in df.columns:
        df = df.dropna(subset=["salary_in_usd"])
    elif "salary" in df.columns:
        df = df.dropna(subset=["salary"])

    # Rename columns to canonical names if they differ
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)

    logger.info("Cleaned salaries: %d rows, %d columns", *df.shape)
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values.
    - `drop`: remove rows with any NaN
    - `fill`: fill numeric with median, categorical with mode
    """
    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with missing values", before - len(df))
    elif strategy == "fill":
        for col in df.columns:
            if df[col].dtype in (np.float64, np.int64):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        logger.info("Filled missing values")

    return df


def remove_outliers(df: pd.DataFrame, col: str, method: str = "iqr") -> pd.DataFrame:
    """
    Remove outliers from a numeric column.
    - `iqr`: use 1.5 × IQR rule
    - `zscore`: remove rows with |z| > 3
    """
    if method == "iqr":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        logger.info("Removed %d salary outliers (IQR method)", before - len(df))
    elif method == "zscore":
        from scipy import stats
        before = len(df)
        z = np.abs(stats.zscore(df[col].dropna()))
        df = df.loc[z[z < 3].index]
        logger.info("Removed %d salary outliers (Z-score method)", before - len(df))

    return df


def extract_skills(df: pd.DataFrame, text_col: str = "job_description") -> pd.DataFrame:
    """
    Basic skill extraction from a text column using keyword matching.
    Expand the SKILLS list with your own keywords.
    """
    SKILLS = [
        "python", "sql", "r", "java", "scala", "aws", "azure", "gcp",
        "tensorflow", "pytorch", "spark", "hadoop", "excel", "tableau",
        "power bi", "docker", "kubernetes", "git", "machine learning",
        "deep learning", "nlp", "computer vision", "statistics",
        "airflow", "dbt", "snowflake", "databricks",
    ]

    if text_col not in df.columns:
        logger.warning("Column '%s' not found — skipping skill extraction.", text_col)
        return df

    text_lower = df[text_col].fillna("").str.lower()
    for skill in SKILLS:
        df[f"skill_{skill.replace(' ', '_')}"] = text_lower.str.contains(skill).astype(int)

    logger.info("Extracted %d skill features", len(SKILLS))
    return df


def encode_categoricals(
    df: pd.DataFrame,
    target_col: str = "salary_in_usd",
    categorical_cols: list | None = None,
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Encode categorical columns with OneHotEncoder and scale numeric columns.
    Returns the transformed array + the fitted preprocessor.
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Don't encode the target
        categorical_cols = [c for c in categorical_cols if c != target_col]

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    # We return the preprocessor to re-use later; the caller can fit_transform
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str = "salary_in_usd",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split returning X_train, X_test, y_train, y_test."""
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
