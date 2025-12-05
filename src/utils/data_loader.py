# src/utils/data_loader.py
import pandas as pd
from typing import Optional

def load_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load CSV into a DataFrame.
    - path: file path
    - parse_dates: list of column names to parse as dates
    """
    df = pd.read_csv(path, low_memory=False)
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
