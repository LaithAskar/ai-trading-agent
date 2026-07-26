from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol, cast

from ..backtest.runner import load_strategy
from ..config import CACHE_DIR
from ..core.events import Bar
from ..core.orders import Order
from ..core.portfolio import Portfolio
from ..data.yfinance_source import iter_bars, load_bars
from .alpaca import AccountSnapshot, AlpacaOrder, AlpacaPosition


PAPER_ORDER_CHECKPOINT_DB = CACHE_DIR / "paper_orders.sqlite3"
PENDING_CLAIM_STALE_AFTER = timedelta(minutes=5)


class PaperBroker(Protocol):
    def account(self) -> AccountSnapshot: ...
    def positions(self) -> list[AlpacaPosition]: ...
    def submit_market_order(
        self, order: Order, *, client_order_id: str | None = None
    ) -> AlpacaOrder: ...


@dataclass(frozen=True)
class _Claim:
    state: str
    owned: bool
    updated_at: datetime


class _PaperOrderCheckpoint:
    """Atomic local claims for paper-order intents.

    A pending row is committed before broker I/O. That makes one process the
    submitter while other processes reconcile or wait. Pending never means the
    broker accepted an order; only a confirmed response/lookup may transition
    it to submitted.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30)
        self._connection.execute("PRAGMA busy_timeout = 30000")
        try:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_order_claims (
                    idempotency_key TEXT PRIMARY KEY,
                    client_order_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'submitted')),
                    broker_order_id TEXT,
                    claimed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Preserve checkpoints made by the first idempotency implementation.
            legacy = self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='submitted_orders'"
            ).fetchone()
            if legacy:
                now = _utcnow().isoformat()
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO paper_order_claims
                        (idempotency_key, client_order_id, state, broker_order_id,
                         claimed_at, updated_at)
                    SELECT idempotency_key, 'paper-' || substr(idempotency_key, 1, 42),
                           'submitted', broker_order_id, submitted_at, ?
                    FROM submitted_orders
                    """,
                    (now,),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            self._connection.close()
            raise

    def claim(self, idempotency_key: str, client_order_id: str) -> _Claim:
        now = _utcnow()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO paper_order_claims
                    (idempotency_key, client_order_id, state, broker_order_id,
                     claimed_at, updated_at)
                VALUES (?, ?, 'pending', NULL, ?, ?)
                """,
                (idempotency_key, client_order_id, now.isoformat(), now.isoformat()),
            )
            row = self._connection.execute(
                "SELECT state, updated_at FROM paper_order_claims WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        assert row is not None
        return _Claim(row[0], cursor.rowcount == 1, datetime.fromisoformat(row[1]))

    def reclaim_if_stale(self, idempotency_key: str, stale_before: datetime) -> bool:
        now = _utcnow().isoformat()
        try:
            cursor = self._connection.execute(
                """
                UPDATE paper_order_claims
                SET claimed_at = ?, updated_at = ?
                WHERE idempotency_key = ? AND state = 'pending' AND updated_at <= ?
                """,
                (now, now, idempotency_key, stale_before.isoformat()),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return cursor.rowcount == 1

    def mark_submitted(self, idempotency_key: str, broker_order_id: str) -> None:
        try:
            cursor = self._connection.execute(
                """
                UPDATE paper_order_claims
                SET state = 'submitted', broker_order_id = ?, updated_at = ?
                WHERE idempotency_key = ?
                """,
                (broker_order_id, _utcnow().isoformat(), idempotency_key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("paper order claim disappeared before checkpoint")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def close(self) -> None:
        self._connection.close()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _paper_order_key(
    *, strategy_name: str, params: dict, requested_symbol: str, bar: Bar,
    order: Order, order_index: int,
) -> str:
    """Hash the stable strategy/bar/order identity used for retry deduplication."""
    identity = {
        "version": 1,
        "strategy": strategy_name,
        "params": params,
        "requested_symbol": requested_symbol.upper(),
        "bar_timestamp": bar.timestamp.isoformat(),
        "order_index": order_index,
        "order": {
            "symbol": order.symbol.upper(),
            "side": order.side.value,
            "quantity": order.quantity,
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _client_order_id(idempotency_key: str) -> str:
    # Alpaca accepts client order IDs up to 48 characters.
    return f"paper-{idempotency_key[:42]}"


@dataclass(frozen=True)
class PaperOrderOutcome:
    order: Order
    client_order_id: str
    status: str
    broker_order: AlpacaOrder | None = None
    detail: str | None = None


@dataclass
class PaperTickResult:
    symbol: str
    strategy: str
    bars_seen: int
    proposed_orders: list[Order]
    submitted: list[AlpacaOrder]
    dry_run: bool
    skipped_reason: str | None
    outcomes: list[PaperOrderOutcome]

    def _count(self, *statuses: str) -> int:
        return sum(outcome.status in statuses for outcome in self.outcomes)

    @property
    def submitted_count(self) -> int:
        return self._count("submitted", "submitted_checkpoint_error")

    @property
    def already_submitted_count(self) -> int:
        return self._count("already_submitted")

    @property
    def reconciled_count(self) -> int:
        return self._count("reconciled", "reconciled_checkpoint_error")

    @property
    def pending_count(self) -> int:
        return self._count("pending")

    @property
    def failed_count(self) -> int:
        return self._count("failed")

    @property
    def checkpoint_error_count(self) -> int:
        return self._count("submitted_checkpoint_error", "reconciled_checkpoint_error")


def _broker_lookup(broker: PaperBroker, client_order_id: str) -> tuple[bool, AlpacaOrder | None, str | None]:
    # Inspect the class, rather than hasattr(instance), so MagicMock-based legacy
    # fixtures are not mistaken for brokers that implement reconciliation.
    lookup = getattr(type(broker), "get_order_by_client_order_id", None)
    if not callable(lookup):
        return False, None, None
    try:
        return True, cast(AlpacaOrder | None, lookup(broker, client_order_id)), None
    except Exception as exc:
        return True, None, f"broker reconciliation lookup failed: {exc}"


def _result(
    *, symbol: str, strategy_name: str, bars_seen: int, proposed: list[Order],
    submitted: list[AlpacaOrder], dry_run: bool,
    outcomes: list[PaperOrderOutcome], skipped_reason: str | None = None,
) -> PaperTickResult:
    if not dry_run and proposed and not submitted and skipped_reason is None:
        counts: dict[str, int] = {}
        for outcome in outcomes:
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
        summary = ", ".join(f"{count} {status.replace('_', ' ')}" for status, count in counts.items())
        skipped_reason = f"no orders submitted: {summary or 'no actionable outcomes'}"
    return PaperTickResult(
        symbol, strategy_name, bars_seen, proposed, submitted, dry_run,
        skipped_reason, outcomes,
    )


def paper_tick(
    *, strategy_name: str, symbol: str, params: dict | None = None,
    lookback_days: int = 365, broker: PaperBroker | None = None,
    dry_run: bool = False,
) -> PaperTickResult:
    """Replay a strategy and optionally submit deterministic, idempotent paper orders."""
    params = params or {}
    account: AccountSnapshot | None = None
    if broker is not None and not dry_run:
        account = broker.account()
        if account.is_paper is not True:
            raise RuntimeError("paper-only execution refused a non-paper broker account")

    strategy = load_strategy(strategy_name, params)
    end_dt = _utcnow().date()
    start_dt = end_dt - timedelta(days=lookback_days)
    bars: list[Bar] = list(iter_bars(symbol, load_bars(symbol, str(start_dt), str(end_dt))))

    if not bars:
        return _result(
            symbol=symbol, strategy_name=strategy_name, bars_seen=0, proposed=[],
            submitted=[], dry_run=dry_run, outcomes=[],
            skipped_reason="no bars in lookback window",
        )

    synthetic = Portfolio(starting_cash=100_000.0)
    if account is not None:
        synthetic.cash = account.cash
        assert broker is not None
        for pos in broker.positions():
            if pos.symbol == symbol.upper():
                synthetic.positions[pos.symbol] = pos.quantity

    strategy.on_start([symbol])
    proposed: list[Order] = []
    for index, bar in enumerate(bars):
        synthetic.mark(bar.symbol, bar.close)
        emitted = strategy.on_bar(bar, synthetic)
        if index == len(bars) - 1:
            proposed.extend(emitted)

    if dry_run or broker is None:
        return _result(
            symbol=symbol, strategy_name=strategy_name, bars_seen=len(bars),
            proposed=proposed, submitted=[], dry_run=True, outcomes=[],
            skipped_reason=None if proposed else "strategy emitted no orders",
        )

    submitted: list[AlpacaOrder] = []
    outcomes: list[PaperOrderOutcome] = []
    checkpoint = _PaperOrderCheckpoint(PAPER_ORDER_CHECKPOINT_DB)
    try:
        for order_index, order in enumerate(proposed):
            key = _paper_order_key(
                strategy_name=strategy_name, params=params, requested_symbol=symbol,
                bar=bars[-1], order=order, order_index=order_index,
            )
            client_id = _client_order_id(key)
            claim = checkpoint.claim(key, client_id)
            if claim.state == "submitted":
                outcomes.append(PaperOrderOutcome(order, client_id, "already_submitted"))
                continue

            owns_submit = claim.owned
            if not owns_submit:
                supported, found, lookup_error = _broker_lookup(broker, client_id)
                if found is not None:
                    try:
                        checkpoint.mark_submitted(key, found.order_id)
                        outcomes.append(PaperOrderOutcome(order, client_id, "reconciled", found))
                    except Exception as exc:
                        outcomes.append(PaperOrderOutcome(
                            order, client_id, "reconciled_checkpoint_error", found,
                            f"confirmed broker order but checkpoint failed: {exc}",
                        ))
                    continue
                if lookup_error:
                    outcomes.append(PaperOrderOutcome(order, client_id, "pending", detail=lookup_error))
                    continue
                stale_before = _utcnow() - PENDING_CLAIM_STALE_AFTER
                if supported and claim.updated_at <= stale_before:
                    owns_submit = checkpoint.reclaim_if_stale(key, stale_before)
                if not owns_submit:
                    detail = (
                        "pending claim not found at broker; waiting for owner or stale timeout"
                        if supported else
                        "pending claim cannot be reconciled because broker lookup is unsupported"
                    )
                    outcomes.append(PaperOrderOutcome(order, client_id, "pending", detail=detail))
                    continue

            try:
                accepted = broker.submit_market_order(order, client_order_id=client_id)
            except Exception as submit_error:
                _, found, lookup_error = _broker_lookup(broker, client_id)
                if found is None:
                    detail = f"broker submit failed: {submit_error}"
                    if lookup_error:
                        detail += f"; {lookup_error}"
                    outcomes.append(PaperOrderOutcome(order, client_id, "failed", detail=detail))
                    continue
                try:
                    checkpoint.mark_submitted(key, found.order_id)
                    outcomes.append(PaperOrderOutcome(
                        order, client_id, "reconciled", found,
                        f"submit response failed but broker confirmed acceptance: {submit_error}",
                    ))
                except Exception as exc:
                    outcomes.append(PaperOrderOutcome(
                        order, client_id, "reconciled_checkpoint_error", found,
                        f"broker confirmed acceptance but checkpoint failed: {exc}",
                    ))
                continue

            submitted.append(accepted)
            try:
                checkpoint.mark_submitted(key, accepted.order_id)
                outcomes.append(PaperOrderOutcome(order, client_id, "submitted", accepted))
            except Exception as exc:
                outcomes.append(PaperOrderOutcome(
                    order, client_id, "submitted_checkpoint_error", accepted,
                    f"broker accepted order but checkpoint failed: {exc}",
                ))
    finally:
        checkpoint.close()

    return _result(
        symbol=symbol, strategy_name=strategy_name, bars_seen=len(bars),
        proposed=proposed, submitted=submitted, dry_run=False, outcomes=outcomes,
        skipped_reason=None if proposed else "strategy emitted no orders",
    )
