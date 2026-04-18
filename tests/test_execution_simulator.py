import pandas as pd

from research_pipeline import simulate_trade_from_bars


def test_simulate_trade_uses_intraday_barriers_for_long_trade():
    bars = pd.DataFrame(
        [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"open": 100.5, "high": 106.0, "low": 100.0, "close": 105.0},
        ]
    )

    result = simulate_trade_from_bars(
        bars=bars,
        direction="UP",
        entry_price=100.0,
        tp_pct=0.05,
        sl_pct=0.02,
        trailing_pct=0.0,
    )

    assert result["exit_reason"] == "TP"
    assert result["exit_price"] == 105.0
