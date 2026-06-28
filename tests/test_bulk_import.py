import pandas as pd
import pytest

from banknifty_bot.data.bulk_import import load_bulk_csv


def test_separate_date_time_short_names(tmp_path):
    # Common vendor shape: separate Date/Time, abbreviated OHLC, 'Vol'.
    p = tmp_path / "feed.csv"
    pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-01"], "Time": ["09:15:00", "09:20:00"],
        "O": [100, 101], "H": [102, 103], "L": [99, 100], "C": [101, 102], "Vol": [500, 600],
    }).to_csv(p, index=False)

    out = load_bulk_csv(p)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert str(out.index.tz) == "Asia/Kolkata"
    assert out["close"].tolist() == [101, 102]


def test_combined_datetime_and_missing_volume(tmp_path):
    p = tmp_path / "feed.csv"
    pd.DataFrame({
        "datetime": ["2025-01-01 09:15:00", "2025-01-01 09:16:00"],
        "open": [1, 2], "high": [3, 4], "low": [0, 1], "close": [2, 3],
    }).to_csv(p, index=False)

    out = load_bulk_csv(p)
    assert (out["volume"] == 0).all()        # volume synthesised when absent
    assert len(out) == 2


def test_col_map_override_and_dayfirst(tmp_path):
    p = tmp_path / "feed.csv"
    pd.DataFrame({
        "date": ["01-02-2025"], "time": ["09:15:00"],
        "open": [1], "high": [3], "low": [0], "ltp": [2], "qty": [10],
    }).to_csv(p, index=False)

    out = load_bulk_csv(p, col_map={"close": "ltp", "volume": "qty"}, dayfirst=True)
    assert out["close"].iloc[0] == 2 and out["volume"].iloc[0] == 10
    assert out.index[0].month == 2 and out.index[0].day == 1   # dayfirst respected


def test_missing_datetime_raises(tmp_path):
    p = tmp_path / "feed.csv"
    pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_bulk_csv(p)
