from __future__ import annotations
from typing import Any, Dict, List
import os
import sys

# Allow running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestConfig, BacktestEngine
from pyrust_bt.data import load_csv_to_bars
from pyrust_bt.strategy import Strategy
from pyrust_bt.analyzers import compute_drawdown_segments, round_trips_from_trades, export_trades_to_csv, factor_backtest


class FactorStrategy(Strategy):
    def __init__(self, window: int = 3, size: float = 1.0) -> None:
        self.window = window
        self.size = size
        self._closes: List[float] = []

    def next(self, bar: Dict[str, Any]) -> Dict[str, Any] | None:
        close = float(bar["close"])  # type: ignore[assignment]
        self._closes.append(close)
        if len(self._closes) < self.window:
            bar["factor"] = None
            return None
        sma = sum(self._closes[-self.window :]) / self.window
        factor = close - sma
        bar["factor"] = factor
        if factor > 0:
            return {"action": "BUY", "type": "market", "size": self.size}
        elif factor < 0:
            return {"action": "SELL", "type": "market", "size": self.size}
        return None


def main() -> None:
    cfg = BacktestConfig(
        start="2016-01-01",
        end="2025-12-31",
        cash=10000.0,
        commission_rate=0.0005,  # 5 bps 手续费
        slippage_bps=2.0,        # 2 bps 滑点
    )
    engine = BacktestEngine(cfg)

    # Prepare data (expects a CSV at examples/data/sample.csv)
    data_path = os.path.join(os.path.dirname(__file__), "data", "sh600000_min.csv")
    bars = load_csv_to_bars(data_path, symbol="SAMPLE")

    strat = FactorStrategy(window=3, size=1.0)
    res = engine.run(strat, bars)

    equity_curve = res["equity_curve"]
    trades = res["trades"]

    segs = compute_drawdown_segments(equity_curve)
    print("Top drawdown segments:")
    for s in segs[:3]:
        print(s)

    rts = round_trips_from_trades(trades)
    print("Round trips (first 3):", rts[:3])

    out_csv = os.path.join(os.path.dirname(__file__), "data", "trades_out.csv")
    export_trades_to_csv(trades, out_csv)
    print("Trades exported to:", out_csv)

    # Factor backtest over bars (factor written in strategy)
    fb = factor_backtest(bars, factor_key="factor", quantiles=3, forward=1)
    print("Factor backtest:", fb)


if __name__ == "__main__":
    main() 