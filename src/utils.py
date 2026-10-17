import pandas as pd
from pathlib import Path

def load_data():
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "raw" / "job_postings_with_salaries.csv"
    return pd.read_csv(path)