"""Loading real price data.

:func:`load_prices` reads daily prices into a wide table (dates x symbols) from:

- a long table such as a Databento OHLCV export (``ts_event``, ``symbol``, ``close``, ...),
  saved as Parquet or CSV, with ``ts_event`` either as a column or as the index;
- a wide table whose first column (or a column named ``date``) holds the dates;
- a directory of such files, which are concatenated (e.g. one file per month).

:func:`prices_to_returns` keeps the assets with enough history, fills short gaps and turns
prices into simple returns.

Databento OHLCV bars are not adjusted for splits or dividends: a split shows up as a large
spurious return. :func:`prices_to_returns` flags returns above ``suspect_threshold`` and can
neutralise them, but adjusted prices (or Databento's corporate actions data) are the proper fix.
Reading Parquet files needs pyarrow (``pip install -e ".[data]"``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMNS = ("ts_event", "date", "datetime", "timestamp", "time", "ts_recv")
SUFFIXES = (".parquet", ".pq", ".csv")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_dates(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_integer_dtype(values):
        stamps = pd.to_datetime(values, unit="ns", utc=True)
    else:
        stamps = pd.to_datetime(values, utc=True)
    return pd.DatetimeIndex(stamps).tz_convert(None).normalize()


def load_prices(
    path: str | Path,
    symbol_col: str = "symbol",
    price_col: str = "close",
    time_col: str | None = None,
) -> pd.DataFrame:
    """Daily prices as a wide DataFrame indexed by date, one column per symbol."""
    source = Path(path)
    if source.is_dir():
        files = sorted(f for f in source.iterdir() if f.suffix.lower() in SUFFIXES)
    else:
        files = [source]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"no price file found at {source}")
    frame = pd.concat([_read_table(f) for f in files])
    if frame.index.name in TIME_COLUMNS:
        frame = frame.reset_index()

    tcol = time_col or next((c for c in TIME_COLUMNS if c in frame.columns), None)
    if symbol_col in frame.columns:
        if tcol is None:
            raise ValueError(f"no time column found among {TIME_COLUMNS}; pass time_col")
        if price_col not in frame.columns:
            raise ValueError(f"price column {price_col!r} not found")
        long = pd.DataFrame(
            {
                "date": _to_dates(frame[tcol]),
                "symbol": frame[symbol_col].astype(str).to_numpy(),
                "price": pd.to_numeric(frame[price_col], errors="coerce").to_numpy(),
            }
        )
        prices = long.pivot_table(index="date", columns="symbol", values="price", aggfunc="last")
    else:
        tcol = tcol or frame.columns[0]
        prices = frame.set_index(tcol)
        prices.index = _to_dates(prices.index.to_series())
        prices = prices.apply(pd.to_numeric, errors="coerce")

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index.name = "date"
    prices.columns = prices.columns.astype(str)
    prices.columns.name = None
    return prices.astype(float)


def prices_to_returns(
    prices: pd.DataFrame,
    min_coverage: float = 0.95,
    max_fill: int = 5,
    suspect_threshold: float = 0.4,
    neutralise_suspects: bool = False,
) -> pd.DataFrame:
    """Simple daily returns of the assets with enough history.

    Args:
        prices: wide price table (dates x symbols).
        min_coverage: minimum share of dates with a price for an asset to be kept.
        max_fill: longest gap (in rows) filled with the previous price.
        suspect_threshold: absolute daily return above which a warning is raised (possible
            split or data error).
        neutralise_suspects: set the flagged returns to zero instead of keeping them.
    """
    coverage = prices.notna().mean()
    kept = prices.loc[:, coverage >= min_coverage].ffill(limit=max_fill)
    returns = (kept / kept.shift(1) - 1.0).iloc[1:].dropna(how="any")
    returns = returns.loc[:, returns.std() > 0]
    if returns.shape[1] < 2:
        raise ValueError("fewer than two usable assets after cleaning")

    suspects = returns.abs() > suspect_threshold
    if suspects.to_numpy().any():
        flagged = returns.where(suspects).stack().dropna()
        examples = ", ".join(f"{sym} {d.date()} ({r:+.0%})" for (d, sym), r in flagged.head(5).items())
        warnings.warn(
            f"{len(flagged)} daily returns above {suspect_threshold:.0%} (e.g. {examples}); "
            "check for unadjusted splits",
            stacklevel=2,
        )
        if neutralise_suspects:
            returns = returns.mask(suspects, 0.0)
    return returns.astype(np.float64)
