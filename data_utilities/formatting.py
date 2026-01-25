# Formatting utilities 
from datetime import date
import pandas as pd
from typing import Optional

def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")

def parse_date(column_name: pd.Series) -> Optional[date]:
    """
        Parse 'YYYY-MM-DD' into datetime.date. 
        Returns None on failure.
    """
    try:
        return pd.to_datetime(column_name, errors="coerce")
    except Exception:
        raise ValueError ("parse_date got an invalid input")
    
def parse_numeric(column_name: pd.Series) -> pd.Series:
    """
        Coerce a Series to numeric.
        Non-parsable values become NaN.
    """
    try:
        return pd.to_numeric(column_name, errors="coerce")
    except Exception:
        raise ValueError ("parse_numeric got an invalid input")
    
def change_column_name(df, list_of_col_names, correct_col_name):
    for col_name in df.columns:
        if col_name in list_of_col_names:
            df = df.rename(columns = {col_name: correct_col_name})
            return df
    return df