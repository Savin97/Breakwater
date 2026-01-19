# Formatting utilities 
from datetime import date
import pandas as pd
from typing import Optional

def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")

def parse_date(x) -> Optional[date]:
    """Parse 'YYYY-MM-DD' into datetime.date. Returns None on failure."""
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        raise ValueError ("parse_date got an invalid input")
    
def change_column_name(df, list_of_col_names, correct_col_name):
    for col_name in df.columns:
        if col_name in list_of_col_names:
            df = df.rename(columns = {col_name: correct_col_name})
            return df
    return df