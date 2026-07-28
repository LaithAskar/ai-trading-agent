from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from .config import DATA_DIR
from .core.orders import Order, Side

EXPERIMENTS_DIR = DATA_DIR / "experiments"

ExperimentMode = Literal["research", "paper", "live"]


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ExperimentContract:
    """Hard boundary for autonomous trading experiments."""

    name: str
    capital_cap_usd: float = 200.0
    duration_days: int = 30
    mode: ExperimentMode = "paper"
    autonomous: bool = True
    trade_approval_required: bool = False
    broker: str = "alpaca"
    allowed_assets: list[str] = field(default_factory=lambda: ["US_EQUITY", "ETF"])
    blocked_assets: list[str] = field(default_factory=lambda: ["OPTIONS", "MARGIN", "SHORTS", "CRYPTO"])
    max_trade_usd: float = 20.0
    max_position_usd: float = 40.0
    daily_loss_stop_usd: float = 4.0
    total_loss_stop_usd: float = 30.0
    max_trades_per_day: int | None = None
    normal_market_hours_only: bool = True
    no_open_close_auction_window_minutes: int = 15
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    live_enabled: bool = False
    notes: str = "Autonomous graph trading v0; max_trades_per_day=None means deterministic strategy/risk gates decide activity."

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("name may only contain letters, numbers, dot, underscore, and dash")
        finite_limits = {
            "capital_cap_usd": self.capital_cap_usd,
            "max_trade_usd": self.max_trade_usd,
            "max_position_usd": self.max_position_usd,
            "daily_loss_stop_usd": self.daily_loss_stop_usd,
            "total_loss_stop_usd": self.total_loss_stop_usd,
        }
        for field_name, value in finite_limits.items():
            if not _is_finite_number(value):
                raise ValueError(f"{field_name} must be finite")
        if self.capital_cap_usd <= 0:
            raise ValueError("capital_cap_usd must be positive")
        if self.duration_days <= 0:
            raise ValueError("duration_days must be positive")
        if self.mode not in {"research", "paper", "live"}:
            raise ValueError("mode must be research, paper, or live")
        if self.mode == "live" and not self.live_enabled:
            raise ValueError("live mode requires live_enabled=True in the contract")
        if self.max_trade_usd <= 0:
            raise ValueError("max_trade_usd must be positive")
        if self.max_position_usd <= 0:
            raise ValueError("max_position_usd must be positive")
        if self.max_trade_usd > self.capital_cap_usd:
            raise ValueError("max_trade_usd cannot exceed capital_cap_usd")
        if self.max_position_usd > self.capital_cap_usd:
            raise ValueError("max_position_usd cannot exceed capital_cap_usd")
        if self.daily_loss_stop_usd <= 0:
            raise ValueError("daily_loss_stop_usd must be positive")
        if self.total_loss_stop_usd <= 0:
            raise ValueError("total_loss_stop_usd must be positive")
        if self.max_trades_per_day is not None and self.max_trades_per_day <= 0:
            raise ValueError("max_trades_per_day must be positive or None")
        if self.no_open_close_auction_window_minutes < 0:
            raise ValueError("no_open_close_auction_window_minutes cannot be negative")
        if not self.allowed_assets:
            raise ValueError("allowed_assets cannot be empty")
        allowed = {value.upper() for value in self.allowed_assets}
        blocked = {value.upper() for value in self.blocked_assets}
        if allowed & blocked:
            raise ValueError("allowed_assets and blocked_assets cannot overlap")

    @property
    def path(self) -> Path:
        return EXPERIMENTS_DIR / f"{self.name}.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self) -> Path:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, name: str) -> "ExperimentContract":
        path = EXPERIMENTS_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"experiment contract not found: {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    portfolio_value: float
    positions_market_value: dict[str, float]
    positions_quantity: dict[str, float] = field(default_factory=dict)
    daily_pnl: float | None = None
    total_pnl: float | None = None
    trades_today: int | None = None
    market_is_open: bool | None = None
    observed_at: datetime | None = None
    session_open: datetime | None = None
    session_close: datetime | None = None
    as_of: str = field(default_factory=lambda: date.today().isoformat())


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str
    order_notional: float
    projected_symbol_exposure: float
    projected_total_exposure: float = 0.0


def _market_boundary_reason(contract: ExperimentContract, portfolio: PortfolioSnapshot) -> str | None:
    if not contract.normal_market_hours_only:
        return None
    if portfolio.market_is_open is None or portfolio.observed_at is None:
        return "market-hours state unavailable"
    if not portfolio.market_is_open:
        return "market is closed"
    if portfolio.session_open is None or portfolio.session_close is None:
        return "market session boundary unavailable"
    observed = portfolio.observed_at
    opened = portfolio.session_open
    closes = portfolio.session_close
    if observed.tzinfo is None or opened.tzinfo is None or closes.tzinfo is None:
        return "market session timestamps must be timezone-aware"
    boundary_seconds = contract.no_open_close_auction_window_minutes * 60
    if (observed - opened).total_seconds() < boundary_seconds:
        return "inside market-open auction boundary"
    if (closes - observed).total_seconds() <= boundary_seconds:
        return "inside market-close auction boundary"
    return None


def evaluate_order(
    contract: ExperimentContract,
    order: Order,
    *,
    price: float,
    portfolio: PortfolioSnapshot,
    asset_class: str,
) -> GateDecision:
    """Return a fail-closed deterministic decision for one proposed order."""
    if not _is_finite_number(price) or price <= 0:
        return GateDecision(False, "price must be finite and positive", 0.0, 0.0, 0.0)

    snapshot_values = [portfolio.cash, portfolio.portfolio_value]
    snapshot_values.extend(portfolio.positions_market_value.values())
    snapshot_values.extend(portfolio.positions_quantity.values())
    if portfolio.daily_pnl is not None:
        snapshot_values.append(portfolio.daily_pnl)
    if portfolio.total_pnl is not None:
        snapshot_values.append(portfolio.total_pnl)
    if any(not _is_finite_number(value) for value in snapshot_values):
        return GateDecision(False, "portfolio snapshot contains non-finite values", 0.0, 0.0, 0.0)
    if portfolio.cash < 0 or portfolio.portfolio_value < 0:
        return GateDecision(False, "portfolio snapshot contains negative account values", 0.0, 0.0, 0.0)
    if portfolio.trades_today is not None and (
        isinstance(portfolio.trades_today, bool)
        or not isinstance(portfolio.trades_today, int)
        or portfolio.trades_today < 0
    ):
        return GateDecision(False, "trade-count state is invalid", 0.0, 0.0, 0.0)

    symbol = order.symbol.upper()
    current_exposure = max(0.0, portfolio.positions_market_value.get(symbol, 0.0))
    current_quantity = max(0.0, portfolio.positions_quantity.get(symbol, 0.0))
    total_exposure = sum(max(0.0, value) for value in portfolio.positions_market_value.values())
    notional = order.quantity * price
    projected = current_exposure + notional if order.side is Side.BUY else max(0.0, current_exposure - notional)
    projected_total = total_exposure - current_exposure + projected

    def reject(reason: str) -> GateDecision:
        return GateDecision(False, reason, notional, projected, projected_total)

    if contract.mode == "research":
        return reject("research mode cannot execute orders")
    if contract.mode == "live" and not contract.live_enabled:
        return reject("live execution not enabled")

    normalized_asset = asset_class.strip().upper() if asset_class else ""
    if not normalized_asset:
        return reject("asset class unavailable")
    if normalized_asset in {value.upper() for value in contract.blocked_assets}:
        return reject("asset class is blocked")
    if normalized_asset not in {value.upper() for value in contract.allowed_assets}:
        return reject("asset class is not allowed")

    boundary_reason = _market_boundary_reason(contract, portfolio)
    if boundary_reason:
        return reject(boundary_reason)
    if portfolio.daily_pnl is None:
        return reject("daily loss state unavailable")
    if portfolio.total_pnl is None:
        return reject("total loss state unavailable")
    if portfolio.daily_pnl <= -contract.daily_loss_stop_usd:
        return reject("daily loss stop reached")
    if portfolio.total_pnl <= -contract.total_loss_stop_usd:
        return reject("total loss stop reached")
    if contract.max_trades_per_day is not None:
        if portfolio.trades_today is None:
            return reject("trade-count state unavailable")
        if portfolio.trades_today >= contract.max_trades_per_day:
            return reject("max trades per day reached")
    if notional > contract.max_trade_usd:
        return reject("trade exceeds max_trade_usd")
    if order.side is Side.SELL and order.quantity > current_quantity:
        return reject("sell would create a short position")
    if order.side is Side.BUY and projected > contract.max_position_usd:
        return reject("symbol exposure exceeds max_position_usd")
    if order.side is Side.BUY and projected_total > contract.capital_cap_usd:
        return reject("projected exposure exceeds capital_cap_usd")
    if order.side is Side.BUY and notional > portfolio.cash:
        return reject("insufficient cash")

    return GateDecision(True, "allowed", notional, projected, projected_total)


def evaluate_order_batch(
    contract: ExperimentContract,
    orders: Sequence[Order],
    *,
    prices: dict[str, float],
    portfolio: PortfolioSnapshot,
    asset_classes: dict[str, str],
) -> list[GateDecision]:
    """Evaluate in order while reserving cash/exposure/trade count cumulatively."""
    working = portfolio
    decisions: list[GateDecision] = []
    for order in orders:
        symbol = order.symbol.upper()
        decision = evaluate_order(
            contract,
            order,
            price=prices.get(symbol, 0.0),
            portfolio=working,
            asset_class=asset_classes.get(symbol, ""),
        )
        decisions.append(decision)
        if not decision.allowed:
            continue
        market_values = dict(working.positions_market_value)
        quantities = dict(working.positions_quantity)
        market_values[symbol] = decision.projected_symbol_exposure
        signed_qty = order.quantity if order.side is Side.BUY else -order.quantity
        quantities[symbol] = quantities.get(symbol, 0.0) + signed_qty
        cash_delta = -decision.order_notional if order.side is Side.BUY else decision.order_notional
        working = replace(
            working,
            cash=working.cash + cash_delta,
            positions_market_value=market_values,
            positions_quantity=quantities,
            trades_today=None if working.trades_today is None else working.trades_today + 1,
        )
    return decisions


def list_contracts() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(EXPERIMENTS_DIR.glob("*.json"))
