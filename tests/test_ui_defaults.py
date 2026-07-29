from __future__ import annotations

from trading_agent.backtest.runner import load_strategy
from trading_agent.ui_defaults import STRATEGY_DEFAULT_PARAMS, strategy_default_params_text


def test_backtest_ui_defaults_are_strategy_specific_and_constructor_valid():
    assert strategy_default_params_text("sma_cross") == "fast=20,slow=50"
    assert "fast" not in STRATEGY_DEFAULT_PARAMS["rsi_mean_rev"]
    assert "period" not in STRATEGY_DEFAULT_PARAMS["sma_cross"]

    for strategy, params in STRATEGY_DEFAULT_PARAMS.items():
        loaded = load_strategy(strategy, params)
        assert loaded.name == strategy


def test_unknown_strategy_has_no_leaked_sma_defaults():
    assert strategy_default_params_text("future_strategy") == ""
