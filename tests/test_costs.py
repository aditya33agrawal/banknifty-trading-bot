import pytest

from banknifty_bot.backtest.costs import CostModel

FUTURES_PARAMS = {
    "brokerage_per_order": 20.0,
    "stt_pct_on_sell_value": 0.02,
    "exchange_txn_pct_on_value": 0.0019,
    "sebi_pct_on_value": 0.0001,
    "stamp_duty_pct_on_buy_value": 0.002,
    "gst_pct": 18.0,
}

OPTIONS_PARAMS = {
    "brokerage_per_order": 20.0,
    "stt_pct_on_sell_premium": 0.1,
    "exchange_txn_pct_on_premium": 0.035,
    "sebi_pct_on_premium": 0.0001,
    "stamp_duty_pct_on_buy_premium": 0.003,
    "gst_pct": 18.0,
}


def test_futures_round_trip_long_hand_computed():
    model = CostModel("futures", FUTURES_PARAMS)
    qty = 15  # 1 lot
    entry_price, exit_price = 50000.0, 50100.0
    entry_value = entry_price * qty  # 750000
    exit_value = exit_price * qty  # 751500

    breakdown = model.round_trip_cost(entry_price, exit_price, qty, "long")

    assert breakdown.brokerage == pytest.approx(40.0)
    assert breakdown.stt == pytest.approx(exit_value * 0.02 / 100)
    assert breakdown.stamp_duty == pytest.approx(entry_value * 0.002 / 100)
    expected_exch = (entry_value + exit_value) * 0.0019 / 100
    assert breakdown.exchange_txn == pytest.approx(expected_exch)
    expected_gst = (breakdown.brokerage + breakdown.exchange_txn + breakdown.sebi) * 0.18
    assert breakdown.gst == pytest.approx(expected_gst)


def test_futures_short_swaps_buy_sell_legs():
    model = CostModel("futures", FUTURES_PARAMS)
    long_bd = model.round_trip_cost(50000.0, 50100.0, 15, "long")
    short_bd = model.round_trip_cost(50000.0, 50100.0, 15, "short")
    # STT applies to the sell leg; for a short, entry is the sell.
    assert short_bd.stt == pytest.approx(50000.0 * 15 * 0.02 / 100)
    assert long_bd.stt == pytest.approx(50100.0 * 15 * 0.02 / 100)


def test_options_premium_basis():
    model = CostModel("options", OPTIONS_PARAMS)
    qty = 15
    breakdown = model.round_trip_cost(100.0, 150.0, qty, "long")
    entry_value, exit_value = 100.0 * qty, 150.0 * qty

    assert breakdown.stt == pytest.approx(exit_value * 0.1 / 100)
    assert breakdown.stamp_duty == pytest.approx(entry_value * 0.003 / 100)


def test_total_sums_all_components():
    model = CostModel("futures", FUTURES_PARAMS)
    bd = model.round_trip_cost(50000.0, 50100.0, 15, "long")
    assert bd.total == pytest.approx(
        bd.brokerage + bd.stt + bd.exchange_txn + bd.sebi + bd.stamp_duty + bd.gst
    )
