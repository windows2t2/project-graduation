"""
visuals/export_for_tableau.py — v4: Export processed data for Tableau dashboards.
Run: python visuals/export_for_tableau.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import DATA_PROCESSED, logger

OUTPUT_DIR = Path(__file__).resolve().parent

def export():
    """Export key datasets for Tableau visualization."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. All Jobs
    all_path = DATA_PROCESSED / "all_jobs.csv"
    if all_path.exists():
        df_all = pd.read_csv(all_path)
        df_all.to_csv(OUTPUT_DIR / "tableau_all_jobs.csv", index=False)
        logger.info("Exported: tableau_all_jobs.csv (%d rows)", len(df_all))

    # 2. DS/ML/DL Jobs
    dsml_path = DATA_PROCESSED / "ds_ml_dl_jobs.csv"
    if dsml_path.exists():
        df_dsml = pd.read_csv(dsml_path)
        df_dsml.to_csv(OUTPUT_DIR / "tableau_ds_ml_dl_jobs.csv", index=False)
        logger.info("Exported: tableau_ds_ml_dl_jobs.csv (%d rows)", len(df_dsml))

    # 3. Salary by Experience Level
    if all_path.exists():
        df_all = pd.read_csv(all_path)
        salary_exp = df_all.groupby("experience_level")["salary_in_usd"].agg(["mean", "median", "count"]).round(0)
        salary_exp.to_csv(OUTPUT_DIR / "tableau_salary_by_experience.csv")
        logger.info("Exported: tableau_salary_by_experience.csv (%d groups)", len(salary_exp))

    # 4. Salary by Job Category (DS/ML/DL only)
    if dsml_path.exists():
        df_dsml = pd.read_csv(dsml_path)
        if "ds_ml_field" in df_dsml.columns:
            salary_cat = df_dsml.groupby("ds_ml_field")["salary_in_usd"].agg(["mean", "median", "count"]).round(0)
            salary_cat.to_csv(OUTPUT_DIR / "tableau_salary_by_category.csv")
            logger.info("Exported: tableau_salary_by_category.csv (%d groups)", len(salary_cat))

    # 5. Remote Work Trends
    if all_path.exists():
        df_all = pd.read_csv(all_path)
        remote_trends = df_all.groupby(["work_year", "remote_ratio"]).size().reset_index(name="count")
        remote_trends.to_csv(OUTPUT_DIR / "tableau_remote_trends.csv", index=False)
        logger.info("Exported: tableau_remote_trends.csv (%d rows)", len(remote_trends))

    print("\n✅ All Tableau exports complete! Files saved to visuals/")

if __name__ == "__main__":
    export()
