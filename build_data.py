"""Build v4 data"""
import sys; sys.path.insert(0, ".")
from src.data_loader import load_and_prepare
df_all, df_dsml = load_and_prepare()
print(f"All jobs: {len(df_all)} rows")
print(f"DS/ML/DL: {len(df_dsml)} rows")
print("\nDS/ML Fields:")
print(df_dsml["ds_ml_field"].value_counts().to_string())
print("\nMedian salary by field:")
print(df_dsml.groupby("ds_ml_field")["salary_in_usd"].median().sort_values(ascending=False).to_string())
