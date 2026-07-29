from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from ..backtest.runner import load_strategy
from ..core.events import Bar
from ..core.orders import Order
from ..core.portfolio import Portfolio
from ..data.yfinance_source import iter_bars, load_bars
from ..experiment import ExperimentContract, GateDecision, PortfolioSnapshot, evaluate_order_batch
from ..experiment_graph import ExperimentGraph
from .alpaca import AlpacaOrder, AlpacaPaperBroker


@dataclass(frozen=True)
class TradeExecutionRecord:
    order: Order
    reference_price: float
    decision: GateDecision
    status: str
    client_order_id: str | None = None
    broker_order_id: str | None = None


@dataclass
class PaperTickResult:
    symbol: str
    strategy: str
    bars_seen: int
    proposed_orders: list[Order]
    submitted: list[AlpacaOrder]
    dry_run: bool
    skipped_reason: str | None
    rejected_orders: list[str]
    execution_records: list[TradeExecutionRecord]


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _client_order_id(contract: ExperimentContract, strategy: str, order: Order, trading_day: str, index: int) -> str:
    payload = json.dumps(
        {
            "experiment": contract.name,
            "strategy": strategy,
            "day": trading_day,
            "index": index,
            "symbol": order.symbol.upper(),
            "side": order.side.value,
            "quantity": order.quantity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "ta-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


def _empty_result(symbol: str, strategy: str, dry_run: bool, reason: str) -> PaperTickResult:
    return PaperTickResult(symbol, strategy, 0, [], [], dry_run, reason, [], [])


def paper_tick(
    *,
    strategy_name: str,
    symbol: str,
    params: dict | None = None,
    lookback_days: int = 365,
    broker: AlpacaPaperBroker | None = None,
    dry_run: bool = False,
    contract: ExperimentContract | None = None,
    graph: ExperimentGraph | None = None,
    run_id: str = "standalone",
) -> PaperTickResult:
    """Replay strategy state, gate its latest batch, and submit only to Alpaca paper."""
    params = params or {}
    strategy = load_strategy(strategy_name, params)

    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=lookback_days)
    df = load_bars(symbol, str(start_dt), str(end_dt))
    bars: list[Bar] = list(iter_bars(symbol, df))
    if not bars:
        return _empty_result(symbol, strategy_name, dry_run, "no bars in lookback window")

    synthetic = Portfolio(starting_cash=contract.max_trade_usd if contract is not None else 100_000.0)
    acct = None
    broker_positions = []
    snapshot_errors: list[str] = []
    if broker is not None and not dry_run:
        try:
            acct = broker.account()
        except Exception as exc:
            snapshot_errors.append(f"account: {type(exc).__name__}: {exc}")
        try:
            broker_positions = broker.positions()
        except Exception as exc:
            snapshot_errors.append(f"positions: {type(exc).__name__}: {exc}")
        if acct is not None:
            account_values = [acct.cash, acct.portfolio_value, acct.buying_power]
            if acct.daily_pnl is not None:
                account_values.append(acct.daily_pnl)
            if any(not _is_finite_number(value) for value in account_values):
                snapshot_errors.append("account: non-finite numeric value")
            else:
                synthetic.cash = min(acct.cash, contract.max_trade_usd) if contract is not None else acct.cash
        for pos in broker_positions:
            if any(
                not _is_finite_number(value)
                for value in (pos.quantity, pos.avg_entry_price, pos.market_value, pos.unrealized_pl)
            ):
                snapshot_errors.append(f"position {pos.symbol}: non-finite numeric value")
                continue
            if pos.symbol.upper() == symbol.upper():
                synthetic.positions[pos.symbol.upper()] = pos.quantity

    strategy.on_start([symbol])
    proposed: list[Order] = []
    for index, bar in enumerate(bars):
        synthetic.mark(bar.symbol, bar.close)
        emitted = strategy.on_bar(bar, synthetic)
        if index == len(bars) - 1:
            proposed.extend(emitted)

    latest_price = bars[-1].close
    if dry_run or broker is None:
        dry_decision = GateDecision(
            False,
            "dry run: contract gate not evaluated without broker snapshot",
            0.0,
            0.0,
            0.0,
        )
        records = [TradeExecutionRecord(order, latest_price, dry_decision, "proposed_dry_run") for order in proposed]
        return PaperTickResult(
            symbol, strategy_name, len(bars), proposed, [], True,
            None if proposed else "strategy emitted no orders", [], records,
        )

    if contract is None:
        return PaperTickResult(
            symbol, strategy_name, len(bars), proposed, [], False,
            "execution requires an experiment contract", [], [],
        )
    graph = graph or ExperimentGraph()
    if acct is None or snapshot_errors:
        reason = "contract gate requires complete broker snapshot: " + "; ".join(snapshot_errors)
        decision = GateDecision(False, reason, 0.0, 0.0, 0.0)
        records = [
            TradeExecutionRecord(order, latest_price, decision, "rejected_invalid_snapshot")
            for order in proposed
        ]
        rejected = [f"{order.symbol} {order.side.value} {order.quantity:g}: invalid broker snapshot" for order in proposed]
        return PaperTickResult(
            symbol, strategy_name, len(bars), proposed, [], False,
            reason, rejected, records,
        )
    if acct.is_paper is not True:
        decision = GateDecision(False, "broker account is not paper", 0.0, 0.0, 0.0)
        records = [
            TradeExecutionRecord(order, latest_price, decision, "rejected_non_paper_account")
            for order in proposed
        ]
        rejected = [f"{order.symbol} {order.side.value} {order.quantity:g}: {decision.reason}" for order in proposed]
        return PaperTickResult(
            symbol,
            strategy_name,
            len(bars),
            proposed,
            [],
            False,
            "execution requires a paper brokerage account",
            rejected,
            records,
        )

    try:
        session = broker.market_session()
    except Exception:
        session = None
    try:
        trades_today = broker.trades_today(None if session is None else session.timestamp)
    except Exception:
        trades_today = None
    asset_classes: dict[str, str] = {}
    for order in proposed:
        try:
            asset_classes[order.symbol.upper()] = broker.asset_class(order.symbol)
        except Exception:
            asset_classes[order.symbol.upper()] = ""

    baseline = graph.baseline_portfolio_value(contract.name, acct.portfolio_value)
    snapshot = PortfolioSnapshot(
        cash=min(acct.cash, contract.capital_cap_usd),
        portfolio_value=acct.portfolio_value,
        positions_market_value={p.symbol.upper(): p.market_value for p in broker_positions},
        positions_quantity={p.symbol.upper(): p.quantity for p in broker_positions},
        daily_pnl=acct.daily_pnl,
        total_pnl=acct.portfolio_value - baseline,
        trades_today=trades_today,
        market_is_open=None if session is None else session.is_open,
        observed_at=None if session is None else session.timestamp,
        session_open=None if session is None else session.session_open,
        session_close=None if session is None else session.session_close,
    )
    prices = {order.symbol.upper(): latest_price if order.symbol.upper() == symbol.upper() else 0.0 for order in proposed}
    decisions = evaluate_order_batch(
        contract,
        proposed,
        prices=prices,
        portfolio=snapshot,
        asset_classes=asset_classes,
    )

    submitted: list[AlpacaOrder] = []
    rejected_orders: list[str] = []
    records: list[TradeExecutionRecord] = []
    trading_day = bars[-1].timestamp.date().isoformat()
    for index, (order, decision) in enumerate(zip(proposed, decisions, strict=True)):
        client_order_id = _client_order_id(contract, strategy_name, order, trading_day, index)
        if not decision.allowed:
            rejected_orders.append(f"{order.symbol} {order.side.value} {order.quantity:g}: {decision.reason}")
            records.append(TradeExecutionRecord(order, latest_price, decision, "rejected", client_order_id))
            continue

        try:
            claim = graph.claim_execution(
                client_order_id=client_order_id,
                experiment_name=contract.name,
                run_id=run_id,
            )
        except Exception as claim_error:
            records.append(
                TradeExecutionRecord(order, latest_price, decision, "checkpoint_error", client_order_id)
            )
            return PaperTickResult(
                symbol, strategy_name, len(bars), proposed, submitted, False,
                f"execution checkpoint failed after {len(submitted)} confirmed submits: "
                f"{type(claim_error).__name__}: {claim_error}",
                rejected_orders,
                records,
            )
        if not claim.acquired:
            if claim.status == "submitted":
                records.append(
                    TradeExecutionRecord(
                        order, latest_price, decision, "idempotent_replay", client_order_id, claim.broker_order_id
                    )
                )
            else:
                records.append(
                    TradeExecutionRecord(order, latest_price, decision, "submission_pending", client_order_id)
                )
            continue

        try:
            response = broker.submit_market_order(order, client_order_id=client_order_id)
        except Exception as submit_error:
            try:
                response = broker.order_by_client_order_id(client_order_id)
            except Exception as lookup_error:
                records.append(
                    TradeExecutionRecord(
                        order,
                        latest_price,
                        decision,
                        "submission_ambiguous",
                        client_order_id,
                    )
                )
                return PaperTickResult(
                    symbol, strategy_name, len(bars), proposed, submitted, False,
                    "broker submission outcome is ambiguous after "
                    f"{len(submitted)} confirmed submits: submit={type(submit_error).__name__}: {submit_error}; "
                    f"lookup={type(lookup_error).__name__}: {lookup_error}",
                    rejected_orders,
                    records,
                )
        submitted.append(response)
        graph.record_execution_success(
            client_order_id=client_order_id,
            experiment_name=contract.name,
            run_id=run_id,
            broker_order_id=response.order_id,
            response=asdict(response),
        )
        records.append(
            TradeExecutionRecord(order, latest_price, decision, "submitted", client_order_id, response.order_id)
        )

    reason = None
    if rejected_orders and not submitted and not any(record.status == "idempotent_replay" for record in records):
        reason = "all proposed orders rejected by contract gate"
    return PaperTickResult(
        symbol, strategy_name, len(bars), proposed, submitted, False, reason, rejected_orders, records
    )
