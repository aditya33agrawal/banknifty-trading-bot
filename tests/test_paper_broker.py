import datetime as dt

import pytest

from banknifty_bot.backtest.costs import CostModel
from banknifty_bot.backtest.slippage import Slippage
from banknifty_bot.paper.paper_broker import PaperBroker

FUTURES_PARAMS = {
    "brokerage_per_order": 20.0,
    "stt_pct_on_sell_value": 0.02,
    "exchange_txn_pct_on_value": 0.0019,
    "sebi_pct_on_value": 0.0001,
    "stamp_duty_pct_on_buy_value": 0.002,
    "gst_pct": 18.0,
}


@pytest.fixture
def broker():
    return PaperBroker(
        CostModel("futures", FUTURES_PARAMS), Slippage("ticks", 0.0),
        contract_multiplier=1.0, initial_equity=1_000_000.0,
    )


def test_open_then_close_records_trade(broker):
    ts1 = dt.datetime(2026, 1, 5, 9, 30)
    ts2 = dt.datetime(2026, 1, 5, 9, 45)

    opened = broker.open_position("long", 15, 50000.0, ts1, stop_loss=49900.0, target=50200.0)
    assert opened
    assert broker.has_open_position()

    broker.close_position(50100.0, ts2, "target")
    assert not broker.has_open_position()
    assert len(broker.portfolio.trades) == 1

    trade = broker.portfolio.trades[0]
    expected_gross = (50100.0 - 50000.0) * 15
    assert trade.gross_pnl == pytest.approx(expected_gross)
    assert trade.net_pnl == pytest.approx(expected_gross - trade.cost)


def test_cannot_open_second_position_while_one_open(broker):
    ts1 = dt.datetime(2026, 1, 5, 9, 30)
    broker.open_position("long", 15, 50000.0, ts1)
    second = broker.open_position("short", 15, 50000.0, ts1)
    assert not second


def test_check_stop_target_detects_stop_hit(broker):
    ts1 = dt.datetime(2026, 1, 5, 9, 30)
    broker.open_position("long", 15, 50000.0, ts1, stop_loss=49900.0, target=50200.0)
    bar = {"high": 50050.0, "low": 49880.0}
    assert broker.check_stop_target(bar) == "stop"
