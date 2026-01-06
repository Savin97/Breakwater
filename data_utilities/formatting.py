# Formatting utilities 
from datetime import date

def today_yyyy_mm_dd() -> str:
    return date.today().strftime("%Y-%m-%d")