"""
src/tavily_job_finder.py — v4: Search for LIVE DS/ML/DL jobs across 6 regions using Tavily.
Saves results to data/processed/live_jobs.csv for the Streamlit Job Finder tab.
"""
import os, json, time
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from tavily import TavilyClient

from src.utils import get_tavily_config, DATA_PROCESSED, logger

# Regions to search + their job boards / queries
REGIONS = {
    "Europe": [
        "data scientist machine learning engineer job 2024 site: linkedin.com europe",
        "AI engineer deep learning job 2024 site: indeed.com europe",
    ],
    "Middle East": [
        "data scientist machine learning job 2024 Dubai UAE Saudi Arabia Qatar",
        "AI engineer job 2024 Middle East site: linkedin.com",
    ],
    "China": [
        "data scientist machine learning engineer job 2024 China Beijing Shanghai",
        "AI deep learning job China 2024 site: zhaopin.com OR site: 51job.com",
    ],
    "Russia": [
        "data scientist machine learning job 2024 Russia Moscow",
        "AI engineer deep learning vacancy Russia 2024",
    ],
    "South America": [
        "data scientist machine learning job 2024 Brazil Argentina Chile Colombia",
        "AI engineer job 2024 South America site: linkedin.com",
    ],
    "East Asia": [
        "data scientist machine learning job 2024 Japan Korea Singapore Taiwan",
        "AI engineer deep learning job 2024 East Asia site: linkedin.com",
    ],
}


def _get_client() -> Optional[TavilyClient]:
    cfg = get_tavily_config()
    if not cfg["api_key"]:
        logger.warning("TAVILY_API_KEY not set — job finder disabled.")
        return None
    return TavilyClient(api_key=cfg["api_key"])


def collect_live_jobs(max_per_query: int = 8) -> pd.DataFrame:
    """
    Search Tavily for DS/ML/DL jobs across all regions.
    Returns a DataFrame with columns: region, title, url, snippet, collected_at
    """
    client = _get_client()
    if not client:
        return pd.DataFrame()

    all_rows = []
    total_queries = sum(len(qs) for qs in REGIONS.values())

    for region, queries in REGIONS.items():
        for q in queries:
            logger.info("Searching %s: %s", region, q[:80])
            try:
                results = client.search(q, max_results=max_per_query, search_depth="advanced")
                for r in results.get("results", []):
                    all_rows.append({
                        "region": region,
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:300],
                        "collected_at": datetime.now().isoformat(),
                    })
                time.sleep(1.2)  # Rate limit
            except Exception as e:
                logger.error("Tavily error for %s: %s", region, e)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["url"])
        out_path = DATA_PROCESSED / "live_jobs.csv"
        df.to_csv(out_path, index=False)
        logger.info("Collected %d unique live jobs → %s", len(df), out_path)
    else:
        logger.warning("No jobs collected. Check TAVILY_API_KEY.")

    return df


def load_live_jobs() -> pd.DataFrame:
    """Load previously collected live jobs, or empty DataFrame."""
    path = DATA_PROCESSED / "live_jobs.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def summary_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Return count of jobs per region."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("region").size().reset_index(name="job_count").sort_values("job_count", ascending=False)


if __name__ == "__main__":
    df = collect_live_jobs()
    if not df.empty:
        print(f"\nCollected {len(df)} live DS/ML/DL jobs:")
        print(summary_by_region(df).to_string(index=False))
    else:
        print("No Tavily key — add TAVILY_API_KEY to .env")
