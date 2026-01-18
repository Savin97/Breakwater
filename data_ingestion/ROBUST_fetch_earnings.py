"""
ROBUST_fetch_earnings.py
Robust earnings date fetcher using multiple providers.
Not used yet in the main pipeline, but kept here for reference.


Fetch historical earnings *report dates* for a list of stocks, going back to
a given start date (default: 2008-01-01).

Primary provider  : Alpha Vantage (function=EARNINGS)
Backup provider   : yfinance (Yahoo Finance)

Usage example
------------
from fetch_earnings import fetch_earnings_for_universe

symbols = ["AAPL", "MSFT", "GOOGL"]
earnings_df = fetch_earnings_for_universe(
    symbols,
    start_date="2008-01-01"
)
print(earnings_df.head())
"""

from __future__ import annotations

import os
import time
import logging
import datetime as dt
from typing import Iterable, List, Optional, Dict

import requests
import pandas as pd

from config import DEFAULT_START_DATE

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic sane default; your main app can override this config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


# Reasonable defaults; tune in your app config if needed
ALPHAVANTAGE_MAX_RETRIES = 3
ALPHAVANTAGE_BACKOFF_SECONDS = 15.0  # on "Note" / rate-limit response
ALPHAVANTAGE_TIMEOUT_SECONDS = 15.0

# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Raised when a data provider fails in a recoverable way."""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _to_date(d: str | dt.date | dt.datetime | pd.Timestamp) -> Optional[dt.date]:
    """Convert various date-like objects to a `datetime.date`.

    Returns None if conversion fails or input is falsy.
    """
    if d is None or d == "":
        return None

    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        return d

    if isinstance(d, (dt.datetime, pd.Timestamp)):
        return d.date()

    # string-like
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None


def _ensure_date(obj: str | dt.date | dt.datetime | None, default: dt.date) -> dt.date:
    """Ensure we have a date, or fall back to default."""
    if obj is None:
        return default
    if isinstance(obj, dt.date) and not isinstance(obj, dt.datetime):
        return obj
    return _to_date(obj) or default


# ---------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------


def _request_alphavantage(
        params: Dict[str, str],
        api_key: str,
        max_retries: int = ALPHAVANTAGE_MAX_RETRIES,
        backoff_seconds: float = ALPHAVANTAGE_BACKOFF_SECONDS,
    ) -> dict:
    """Low-level Alpha Vantage request with simple retry/backoff on rate-limit.

    Raises ProviderError if we exhaust retries or get an error response.
    """
    params = dict(params)  # copy
    params["apikey"] = api_key

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                ALPHAVANTAGE_BASE_URL,
                params=params,
                timeout=ALPHAVANTAGE_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Alpha Vantage request error on attempt %d: %s", attempt, exc
            )
            if attempt == max_retries:
                raise ProviderError(f"Network error contacting Alpha Vantage: {exc}")
            time.sleep(backoff_seconds * attempt)
            continue

        if resp.status_code != 200:
            msg = f"HTTP {resp.status_code} from Alpha Vantage: {resp.text[:200]}"
            logger.warning(msg)
            if attempt == max_retries:
                raise ProviderError(msg)
            time.sleep(backoff_seconds * attempt)
            continue

        try:
            data = resp.json()
        except ValueError:
            msg = "Alpha Vantage response is not valid JSON"
            logger.warning(msg)
            if attempt == max_retries:
                raise ProviderError(msg)
            time.sleep(backoff_seconds * attempt)
            continue

        # Handle rate-limit / generic notes
        if "Note" in data:
            note = data["Note"]
            logger.warning(
                "Alpha Vantage rate limit / note on attempt %d: %s", attempt, note
            )
            if attempt == max_retries:
                raise ProviderError(f"Alpha Vantage rate limited: {note}")
            time.sleep(backoff_seconds * attempt)
            continue

        if "Error Message" in data:
            msg = data["Error Message"]
            raise ProviderError(f"Alpha Vantage error: {msg}")

        return data

    # If we somehow exit the loop without returning
    raise ProviderError("Alpha Vantage: exhausted retries without success")


def fetch_earnings_alphavantage(
        symbol: str,
        start_date: str | dt.date | dt.datetime = DEFAULT_START_DATE,
        api_key: Optional[str] = None,
    ) -> pd.DataFrame:
    """Fetch historical earnings dates for one symbol from Alpha Vantage.

    Uses the `EARNINGS` function (annual + quarterly). We keep only quarterly
    earnings and pull their `reportedDate` as the earnings *announcement*
    date.

    Returns a DataFrame with columns:
    - stock
    - earnings_date
    - fiscal_date_ending
    - reported_eps
    - estimated_eps
    - surprise
    - surprise_pct
    - provider
    """
    if api_key is None:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        raise ProviderError(
            "Alpha Vantage API key not provided. "
            "Pass api_key=... or set ALPHAVANTAGE_API_KEY env var."
        )

    start_date = _ensure_date(start_date, DEFAULT_START_DATE)

    params = {
        "function": "EARNINGS",
        "symbol": symbol,
    }

    data = _request_alphavantage(params=params, api_key=api_key)

    quarterly = data.get("quarterlyEarnings") or []
    if not quarterly:
        # This is not an error, just "no data for this symbol"
        logger.info("Alpha Vantage: no quarterly earnings for %s", symbol)
        return pd.DataFrame(
            columns=[
                "stock",
                "earnings_date",
                "fiscal_date_ending",
                "reported_eps",
                "estimated_eps",
                "surprise",
                "surprise_pct",
                "provider",
            ]
        )

    rows = []
    for q in quarterly:
        fiscal = _to_date(q.get("fiscalDateEnding"))
        reported = _to_date(q.get("reportedDate"))

        if reported is None:
            # If reportedDate missing, skip; it's useless for event timing
            continue

        if reported < start_date:
            continue

        rows.append(
            {
                "stock": symbol,
                "earnings_date": reported,
                "fiscal_date_ending": fiscal,
                "reported_eps": _safe_float(q.get("reportedEPS")),
                "estimated_eps": _safe_float(q.get("estimatedEPS")),
                "surprise": _safe_float(q.get("surprise")),
                "surprise_pct": _safe_float(q.get("surprisePercentage")),
                "provider": "alphavantage",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "stock",
                "earnings_date",
                "fiscal_date_ending",
                "reported_eps",
                "estimated_eps",
                "surprise",
                "surprise_pct",
                "provider",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["stock", "earnings_date"]).reset_index(drop=True)
    return df


def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------
# yfinance backup provider
# ---------------------------------------------------------------------


def fetch_earnings_yfinance(
    symbol: str,
    start_date: str | dt.date | dt.datetime = DEFAULT_START_DATE,
) -> pd.DataFrame:
    """Backup: fetch historical earnings dates using yfinance.

    This relies on `Ticker.get_earnings_dates(limit=...)` and is **best-effort**:
    Yahoo occasionally changes formats, so this function is written defensively.

    Returns the same schema as fetch_earnings_alphavantage, but with fewer
    financial fields (EPS, surprise etc might not be available or consistent).
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ProviderError(
            "yfinance is not installed. Install with `pip install yfinance` "
            "to use the backup provider."
        ) from exc

    start_date = _ensure_date(start_date, DEFAULT_START_DATE)

    ticker = yf.Ticker(symbol)
    try:
        # Use a large limit to get as far back as possible (Yahoo caps at 100)
        df_raw = ticker.get_earnings_dates(limit=100)
    except Exception as exc:
        raise ProviderError(f"yfinance.get_earnings_dates failed for {symbol}: {exc}")

    if df_raw is None or df_raw.empty:
        logger.info("yfinance: no earnings dates for %s", symbol)
        return pd.DataFrame(
            columns=[
                "stock",
                "earnings_date",
                "fiscal_date_ending",
                "reported_eps",
                "estimated_eps",
                "surprise",
                "surprise_pct",
                "provider",
            ]
        )

    # Try to infer the date column/index robustly
    if isinstance(df_raw.index, pd.DatetimeIndex):
        dates = df_raw.index.to_series()
    elif "Earnings Date" in df_raw.columns:
        dates = pd.to_datetime(df_raw["Earnings Date"], errors="coerce")
    else:
        # Fallback: try first column as dates
        first_col = df_raw.columns[0]
        dates = pd.to_datetime(df_raw[first_col], errors="coerce")

    dates = dates.apply(_to_date)
    mask = dates.notna() & dates.apply(lambda d: d >= start_date)
    dates = dates[mask]

    if dates.empty:
        return pd.DataFrame(
            columns=[
                "stock",
                "earnings_date",
                "fiscal_date_ending",
                "reported_eps",
                "estimated_eps",
                "surprise",
                "surprise_pct",
                "provider",
            ]
        )

    rows = []
    for d in dates:
        rows.append(
            {
                "stock": symbol,
                "earnings_date": d,
                "fiscal_date_ending": None,  # not easily available here
                "reported_eps": None,
                "estimated_eps": None,
                "surprise": None,
                "surprise_pct": None,
                "provider": "yfinance",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["stock", "earnings_date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Orchestration: try Alpha Vantage, fall back to backups
# ---------------------------------------------------------------------


def fetch_earnings_for_symbol(
        symbol: str,
        start_date: str | dt.date | dt.datetime = DEFAULT_START_DATE,
        primary: str = "alphavantage",
        backups: Optional[List[str]] = None,
        api_keys: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
    """Fetch earnings dates for a single symbol, trying providers in order.

    Parameters
    ----------
    symbol : str
        Ticker symbol (matching what your engine uses).
    start_date : str | date | datetime
        Earliest earnings *report date* to keep.
    primary : {"alphavantage", "yfinance"}
        First provider to try.
    backups : list[str] or None
        List of backup providers to try if primary fails or returns no rows.
        Example: ["yfinance"]
    api_keys : dict or None
        Optional mapping of provider -> api_key. For Alpha Vantage, key name
        is "alphavantage". If None, we fall back to environment variables.

    Returns
    -------
    DataFrame with unified schema. Can be empty if no data from any provider.
    """
    providers_order: List[str] = [primary]
    if backups:
        for b in backups:
            if b not in providers_order:
                providers_order.append(b)

    api_keys = api_keys or {}

    last_error: Optional[Exception] = None

    for provider in providers_order:
        try:
            if provider == "alphavantage":
                df = fetch_earnings_alphavantage(
                    symbol=symbol,
                    start_date=start_date,
                    api_key=api_keys.get("alphavantage"),
                )
            elif provider == "yfinance":
                df = fetch_earnings_yfinance(symbol=symbol, start_date=start_date)
            else:
                logger.warning("Unknown provider '%s' – skipping", provider)
                continue

            if df is not None and not df.empty:
                logger.info(
                    "Fetched %d earnings rows for %s from %s",
                    len(df),
                    symbol,
                    provider,
                )
                return df

        except ProviderError as exc:
            logger.warning(
                "Provider %s failed for %s with error: %s",
                provider,
                symbol,
                exc,
            )
            last_error = exc
            continue

    # If we reached here, either everything failed or returned empty
    if last_error:
        logger.warning(
            "All providers failed for %s. Last error: %s", symbol, last_error
        )

    return pd.DataFrame(
        columns=[
            "stock",
            "earnings_date",
            "fiscal_date_ending",
            "reported_eps",
            "estimated_eps",
            "surprise",
            "surprise_pct",
            "provider",
        ]
    )


def fetch_earnings_for_universe(
        symbols: Iterable[str],
        start_date: str | dt.date | dt.datetime = DEFAULT_START_DATE,
        primary: str = "alphavantage",
        backups: Optional[List[str]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        deduplicate: bool = True,
        sleep_between_symbols: float = 0.0,
    ) -> pd.DataFrame:
    """Fetch earnings dates for many symbols and stack into one DataFrame.

    Parameters
    ----------
    symbols : iterable of str
        Universe of tickers (e.g. S&P 500).
    start_date : str | date | datetime
        Earliest earnings *report date* to keep.
    primary : {"alphavantage", "yfinance"}
        First provider to try for each symbol.
    backups : list[str] or None
        Providers to try if primary fails or returns empty.
    api_keys : dict or None
        Optional mapping provider -> api_key.
    deduplicate : bool
        If True, drop duplicate (stock, earnings_date).
    sleep_between_symbols : float
        Optional sleep between symbols to be nice to the APIs.

    Returns
    -------
    DataFrame with rows for all symbols that had data.
    """
    frames: List[pd.DataFrame] = []
    symbols = list(symbols)

    logger.info(
        "Fetching earnings for %d symbols (primary=%s, backups=%s)...",
        len(symbols),
        primary,
        backups,
    )

    for i, symbol in enumerate(symbols, start=1):
        logger.info("(%d/%d) Fetching earnings for %s", i, len(symbols), symbol)
        df_symbol = fetch_earnings_for_symbol(
            symbol=symbol,
            start_date=start_date,
            primary=primary,
            backups=backups,
            api_keys=api_keys,
        )

        if df_symbol is not None and not df_symbol.empty:
            frames.append(df_symbol)

        if sleep_between_symbols > 0:
            time.sleep(sleep_between_symbols)

    if not frames:
        logger.warning("No earnings data fetched for the given universe.")
        return pd.DataFrame(
            columns=[
                "stock",
                "earnings_date",
                "fiscal_date_ending",
                "reported_eps",
                "estimated_eps",
                "surprise",
                "surprise_pct",
                "provider",
            ]
        )

    all_df = pd.concat(frames, ignore_index=True)

    # Standardize types
    all_df["earnings_date"] = pd.to_datetime(all_df["earnings_date"]).dt.date
    if "fiscal_date_ending" in all_df.columns:
        all_df["fiscal_date_ending"] = all_df["fiscal_date_ending"].apply(_to_date)

    if deduplicate:
        all_df = (
            all_df.sort_values(["stock", "earnings_date", "provider"])
            .drop_duplicates(subset=["stock", "earnings_date"], keep="first")
            .reset_index(drop=True)
        )

    return all_df


# ---------------------------------------------------------------------
# CLI / quick manual test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check: python fetch_earnings.py AAPL MSFT
    import sys

    tickers = sys.argv[1:] or ["AAPL"]
    df_test = fetch_earnings_for_universe(
        tickers,
        start_date="2008-01-01",
        primary="alphavantage",
        backups=["yfinance"],
    )
    print(df_test.head())
    #df_test.to_csv("earnings_test_output.csv", index=False)
    print(f"Total rows: {len(df_test)}")
