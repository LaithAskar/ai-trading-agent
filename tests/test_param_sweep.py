"""Tests for the param-sweep grid expansion."""
from __future__ import annotations

import pytest
import typer

from trading_agent.cli import _expand_grid


def test_expand_grid_empty_yields_single_empty_dict():
    assert _expand_grid([]) == [{}]


def test_expand_grid_single_axis():
    combos = _expand_grid(["fast=10,20,30"])
    assert combos == [{"fast": 10}, {"fast": 20}, {"fast": 30}]


def test_expand_grid_cartesian_product():
    combos = _expand_grid(["fast=10,20", "slow=50,100,200"])
    assert len(combos) == 6
    fast_vals = sorted({c["fast"] for c in combos})
    slow_vals = sorted({c["slow"] for c in combos})
    assert fast_vals == [10, 20]
    assert slow_vals == [50, 100, 200]


def test_expand_grid_coerces_int_first_then_float():
    combos = _expand_grid(["threshold=0.05,0.10,0.15"])
    assert all(isinstance(c["threshold"], float) for c in combos)
    assert combos[0]["threshold"] == pytest.approx(0.05)


def test_expand_grid_falls_back_to_string():
    combos = _expand_grid(["form=10-Q,10-K"])
    assert combos == [{"form": "10-Q"}, {"form": "10-K"}]


def test_expand_grid_bad_syntax_raises():
    with pytest.raises(typer.BadParameter):
        _expand_grid(["no_equals"])
