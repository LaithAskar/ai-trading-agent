# Strategy provenance: `ai_oligopoly_leaders`

## Status

Independent, rules-based adaptation for research and paper evaluation. It is not an exact reproduction of Belcourt's Autopilot strategies, holdings, allocation process, or implementation, and it is not endorsed by Brooker Belcourt or Autopilot.

## Public source

- Autopilot, “Meet the Pilot, Ep. 2: Brooker Belcourt, Ex-Coatue, Tiger & Citadel, Now on Autopilot”: https://www.youtube.com/watch?v=GAMOPNqu8aU
- Public themes visible in the interview and its chapter index include the “number-one-player” rule, businesses whose brands become verbs, and “AI Leaders & Oligopolies.”

No public source inspected disclosed exact holdings, weights, rebalance timing, entry/exit equations, or risk limits. Autopilot appears to be a closed commercial product; no Brooker Belcourt or official Autopilot source repository or reusable software license was identified.

## License and attribution

No third-party source code was copied or imported. The implementation is original project code and remains under this repository's license. This document provides conceptual attribution; it does not imply that the public investment themes themselves are licensed software.

## Deterministic adaptation

The project converts the broad public themes into testable rules:

1. Operate only on an explicit, user-reviewable leader universe. The default research universe is `NVDA, MSFT, GOOGL, AMZN, META, AVGO, TSM, ASML`; these are project research choices, not claimed Belcourt holdings.
2. Require the latest completed close to be above the slow simple moving average.
3. Require the fast simple moving average to be above the slow average.
4. Require closed-bar momentum over a configured lookback to meet a minimum threshold.
5. Enter long only; never short, use margin, options, or crypto.
6. Size the signal quantity from the lesser of configured `target_notional` and available cash. The research default targets $20 at the completed signal bar's close; it is not a hard next-bar fill cap.
7. Do not add while already holding the symbol.
8. Exit the full position when the close falls below the slow average or the fast average falls below the slow average.

Orders generated from bar `t` fill no earlier than bar `t+1` open under the existing backtest engine. The strategy has no external data dependency beyond the project's existing closed OHLCV bars.

### Execution-price boundary

The next bar can open above the signal close, and market-order slippage can increase the final notional further. A $20 signal-price target can therefore fill above $20. No fixed sizing buffer can guarantee the cap across an overnight gap. This strategy is not eligible for paper or live submission until an execution design enforces the contract with hard-notional orders or a bounded-price mechanism and receives separate adversarial review. Research backtests intentionally preserve next-open fills so this risk remains visible rather than being hidden.

## Promotion boundary

Registration does not make the strategy part of the active autonomous soak. It must first:

- produce reproducible backtest artifacts;
- pass finite-metric validation and the existing statistical candidate gate;
- remain subject to the same $200 experiment contract and deterministic paper risk gate;
- remain excluded from execution until the execution-price boundary above is solved;
- allow no-trade outcomes;
- enter a clearly versioned strategy-set boundary so earlier soak runs are not silently reinterpreted.
