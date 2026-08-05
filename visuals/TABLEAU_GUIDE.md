# 📊 Tableau Visualization Guide — v4

## Exported Data Files

Run the export script to generate the CSV files:

```bash
python visuals/export_for_tableau.py
```

### Generated Files

| File | Description |
|---|---|
| `tableau_all_jobs.csv` | Complete job dataset (all roles) |
| `tableau_ds_ml_dl_jobs.csv` | DS / ML / DL subset with field categorization |
| `tableau_salary_by_experience.csv` | Aggregated salary stats by experience level |
| `tableau_salary_by_category.csv` | Aggregated salary stats by DS/ML field |
| `tableau_remote_trends.csv` | Remote work trends over time |

### Suggested Dashboards

1. **Salary Overview** — Bar charts by experience, company size, and location
2. **DS/ML Field Deep Dive** — Salary comparison across Data Science, ML, DL, NLP, etc.
3. **Remote Work Trends** — Line chart of remote/hybrid/on-site over years
4. **Geographic Heat Map** — Median salary by country
5. **Role Explorer** — Top job titles with salary ranges
