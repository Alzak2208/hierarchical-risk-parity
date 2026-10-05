import warnings

import numpy as np
import pandas as pd
import pytest

from src.data import load_prices, prices_to_returns


def _databento_like_frame() -> pd.DataFrame:
    """Long table with the columns of a Databento OHLCV-1d export (ts_event as the index)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    rows = []
    for symbol, start in (("AAPL", 180.0), ("MSFT", 370.0)):
        for i, ts in enumerate(dates):
            price = start * (1 + 0.01 * i)
            rows.append(
                {
                    "ts_event": ts,
                    "rtype": 35,
                    "publisher_id": 1,
                    "instrument_id": 1 if symbol == "AAPL" else 2,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000,
                    "symbol": symbol,
                }
            )
    return pd.DataFrame(rows).set_index("ts_event")


def test_load_databento_parquet_export(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "ohlcv.parquet"
    _databento_like_frame().to_parquet(path)
    prices = load_prices(path)
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert len(prices) == 5 and prices.index.tz is None
    assert prices.iloc[-1]["AAPL"] == pytest.approx(180.0 * 1.04)


def test_load_a_directory_of_csv_files(tmp_path):
    frame = _databento_like_frame().reset_index()
    frame.iloc[:5].to_csv(tmp_path / "part1.csv", index=False)
    frame.iloc[5:].to_csv(tmp_path / "part2.csv", index=False)
    prices = load_prices(tmp_path)
    assert prices.shape == (5, 2)


def test_load_wide_prices(tmp_path):
    wide = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3), "X": [1.0, 1.1, 1.2], "Y": [2.0, 2.0, 2.2]}
    )
    wide.to_csv(tmp_path / "wide.csv", index=False)
    prices = load_prices(tmp_path / "wide.csv")
    assert list(prices.columns) == ["X", "Y"] and prices.index.name == "date"


def test_prices_to_returns_cleans_the_universe():
    index = pd.date_range("2024-01-01", periods=50, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 50)),
            "B": 50 * np.cumprod(1 + rng.normal(0, 0.01, 50)),
            "FLAT": np.full(50, 10.0),  # zero variance: dropped
            "SPARSE": np.where(np.arange(50) < 10, 5.0, np.nan),  # too little history: dropped
        },
        index=index,
    )
    prices.iloc[20, 0] = np.nan  # a one-day gap is filled
    returns = prices_to_returns(prices)
    assert list(returns.columns) == ["A", "B"]
    assert len(returns) == 49 and not returns.isna().any().any()


def test_suspect_returns_are_flagged_and_can_be_neutralised():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame(
        {"A": [100, 101, 50.5, 51, 52, 51.0], "B": [10, 10.1, 10.2, 10.1, 10.3, 10.2]}, index=index
    )
    with pytest.warns(UserWarning, match=r"^1 daily returns above 40% \(e\.g\. A 2024-01-03 \(-50%\)\)"):
        returns = prices_to_returns(prices, neutralise_suspects=True)
    assert returns.loc[index[2], "A"] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = prices_to_returns(prices)
    assert raw.loc[index[2], "A"] == pytest.approx(-0.5)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "missing.parquet")
