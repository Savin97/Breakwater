import time
import os
from pathlib import Path
from dotenv import load_dotenv

from config import BACKOFF_SECONDS

def sleep_backoff(attempt: int) -> None:
    # exponential backoff with light jitter
    wait = BACKOFF_SECONDS * (2 ** attempt)
    wait = wait + (0.1 * wait)
    print(f"Waiting {wait} seconds")
    time.sleep(wait)

def get_alpha_vantage_api_key() -> str:
    load_dotenv() 
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY environment variable"
        )
    return api_key


def chunk_list(items: list[str], n: int):
    """
        Takes a list and returns it in chunks of size n.
        Example:
        items = ["A", "B", "C", "D", "E", "F", "G"]
        n = 3

        Output (one chunk at a time):
        ["A", "B", "C"]
        ["D", "E", "F"]
        ["G"]

        yield makes this a generator, not a normal function.
        That means:
            It does not return everything at once
            It returns one chunk at a time
            Memory-efficient
            Perfect for large lists (like hundreds of stocks)
    """
    for i in range(0, len(items), n):
        yield items[i:i+n]