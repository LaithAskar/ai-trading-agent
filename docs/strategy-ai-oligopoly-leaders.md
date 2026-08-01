# Strategy provenance: `ai_oligopoly_leaders`

## Status

Independent, rules-based adaptation for research and paper evaluation. It is not Brooker Belcourt's private portfolio, not a copy of Autopilot's implementation, and not endorsed by Brooker Belcourt or Autopilot.

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
6. Size entry notional to the lesser of configured `target_notional` and available cash. The experiment default is exactly $20, matching the existing per-trade cap.
7. Do not add while already holding the symbol.
8. Exit the full position when the close falls below the slow average or the fast average falls below the slow average.

Orders generated from bar `t` fill no earlier than bar `t+1` open under the existing backtest engine. The strategy has no external data dependency beyond the project's existing closed OHLCV bars.

## Promotion boundary

Registration does not make the strategy part of the active autonomous soak. It must first:

- produce reproducible backtest artifacts;
- pass finite-metric validation and the existing statistical candidate gate;
- remain subject to the same $200 experiment contract and deterministic paper risk gate;
- allow no-trade outcomes;
- enter a clearly versioned strategy-set boundary so earlier soak runs are not silently reinterpreted.
