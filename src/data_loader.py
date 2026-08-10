"""
src/data_loader.py — v4: Load, prepare BOTH datasets (All Jobs + DS/ML/DL subset)
"""
import pandas as pd
from src.utils import DATA_RAW, DATA_PROCESSED, logger

DS_ML_DL_KEYWORDS = [
    "data scientist", "machine learning", "deep learning",
    "ml engineer", "dl engineer", "ai engineer", "nlp",
    "computer vision", "data science", "research scientist",
    "mlops", "data engineer",
]

EXPERIENCE_LEVEL_MAP = {
    "EN": "Entry Level",
    "MI": "Mid Level",
    "SE": "Senior Level",
    "EX": "Executive Level",
}

COUNTRY_MAP = {
    "AE": "United Arab Emirates", "AS": "American Samoa", "AT": "Austria",
    "AU": "Australia", "BE": "Belgium", "BR": "Brazil", "CA": "Canada",
    "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "DZ": "Algeria",
    "EE": "Estonia", "ES": "Spain", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "HN": "Honduras", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IL": "Israel", "IN": "India", "IQ": "Iraq",
    "IR": "Iran", "IT": "Italy", "JP": "Japan", "KE": "Kenya",
    "LU": "Luxembourg", "MD": "Moldova", "MT": "Malta", "MX": "Mexico",
    "MY": "Malaysia", "NG": "Nigeria", "NL": "Netherlands", "NZ": "New Zealand",
    "PK": "Pakistan", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RU": "Russia", "SG": "Singapore", "SI": "Slovenia", "TR": "Turkey",
    "TN": "Tunisia",
    "UA": "Ukraine", "US": "United States", "VN": "Vietnam",
}

def load_and_prepare():
    """Load raw data, clean, return (df_all, df_ds_ml_dl)."""
    path = DATA_RAW / "ds_salaries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Clean — drop duplicates, drop rows without salary
    df = df.drop_duplicates()
    if "salary_in_usd" in df.columns:
        df = df.dropna(subset=["salary_in_usd"])

    # Outliers
    Q1 = df["salary_in_usd"].quantile(0.25)
    Q3 = df["salary_in_usd"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["salary_in_usd"] >= Q1 - 1.5 * IQR) & (df["salary_in_usd"] <= Q3 + 1.5 * IQR)]

    # Fill missing
    for col in df.columns:
        if df[col].dtype in ("float64", "int64"):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")

    logger.info("All jobs cleaned: %d rows", len(df))

    # Map abbreviated codes to full names
    if "experience_level" in df.columns:
        df["experience_level"] = df["experience_level"].map(EXPERIENCE_LEVEL_MAP).fillna(df["experience_level"])
    if "company_location" in df.columns:
        df["company_location"] = df["company_location"].map(COUNTRY_MAP).fillna(df["company_location"])
    if "employee_residence" in df.columns:
        df["employee_residence"] = df["employee_residence"].map(COUNTRY_MAP).fillna(df["employee_residence"])

    # DS/ML/DL subset
    mask = df["job_title"].str.lower().apply(
        lambda t: any(kw in str(t) for kw in DS_ML_DL_KEYWORDS)
    )
    df_ds_ml_dl = df[mask].copy()

    def categorize(title):
        t = str(title).lower()
        if "deep learning" in t or "dl" in t.split(): return "Deep Learning"
        if "machine learning" in t or "ml" in t.split(): return "Machine Learning"
        if "nlp" in t or "computer vision" in t: return "AI Specialized"
        if "data scientist" in t or "data science" in t: return "Data Science"
        if "data engineer" in t: return "Data Engineering"
        if "mlops" in t or "ai engineer" in t: return "MLOps / AI"
        if "research" in t: return "Research"
        return "Other DS/ML"

    df_ds_ml_dl["ds_ml_field"] = df_ds_ml_dl["job_title"].apply(categorize)
    logger.info("DS/ML/DL subset: %d rows in %d fields", len(df_ds_ml_dl), df_ds_ml_dl["ds_ml_field"].nunique())

    # Save both
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "all_jobs.csv", index=False)
    df_ds_ml_dl.to_csv(DATA_PROCESSED / "ds_ml_dl_jobs.csv", index=False)
    logger.info("Saved all_jobs.csv (%d rows) + ds_ml_dl_jobs.csv (%d rows)", len(df), len(df_ds_ml_dl))

    return df, df_ds_ml_dl

def load_all_jobs():
    return pd.read_csv(DATA_PROCESSED / "all_jobs.csv")

def load_ds_ml_dl_jobs():
    return pd.read_csv(DATA_PROCESSED / "ds_ml_dl_jobs.csv")
