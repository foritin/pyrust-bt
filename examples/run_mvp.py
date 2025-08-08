from __future__ import annotations
from typing import Any, Dict, List

import os
import sys

# Allow running from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))

from pyrust_bt.api import BacktestEngine, BacktestConfig
from pyrust_bt.strategy import Strategy
from pyrust_bt.data import load_csv_to_bars


class DummySMAStrategy(Strategy):
    def __init__(self, window: int = 5, size: float = 1.0) -> None:
        self.window = window
        self.size = size
        self._closes: List[float] = []

    def next(self, bar: Dict[str, Any]) -> str | Dict[str, Any] | None:
        close = float(bar["close"])  # type: ignore[assignment]
        self._closes.append(close)
        if len(self._closes) < self.window:
            return None
        sma = sum(self._closes[-self.window :]) / self.window
        if close > sma:
            # 市价买入
            return {"action": "BUY", "type": "market", "size": self.size}
        elif close < sma:
            # 限价卖出（用当前价作为限价，示例可立即成交）
            return {"action": "SELL", "type": "limit", "size": self.size, "price": close}
        return None

    def on_order(self, event: Dict[str, Any]) -> None:
        # print("on_order:", event)
        pass

    def on_trade(self, event: Dict[str, Any]) -> None:
        # print("on_trade:", event)
        pass

def main() -> None:
    # Prepare config with commission and slippage
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
    if not os.path.exists(data_path):
        print(f"Sample data not found: {data_path}\nPlease place a CSV with columns: datetime,open,high,low,close,volume")
        sys.exit(1)

    bars = load_csv_to_bars(data_path, symbol="SAMPLE")

    # Run
    strategy = DummySMAStrategy(window=5, size=1.0)
    result = engine.run(strategy, bars)

    print("Result:", {k: result[k] for k in ("cash", "position", "avg_cost", "equity", "realized_pnl")})
    print("Stats:", result.get("stats"))


if __name__ == "__main__":
    main() 