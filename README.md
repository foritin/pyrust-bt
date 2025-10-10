# pyrust-bt

A hybrid backtesting framework: Python for strategy and data, Rust for the high-performance core via PyO3 bindings. Designed to balance researcher productivity with engine throughput, suitable from research to small-team production.

[中文文档 | Chinese README](README.zh-CN.md)

## Features
- Rust Engine
  - Time advancement over bars/ticks
  - Order matching: market/limit (same-bar simplified execution)
  - Cost model: commission `commission_rate`, slippage `slippage_bps`
  - Portfolio & ledger: `position / avg_cost / cash / equity / realized_pnl`
  - Vectorized indicators: `SMA / RSI` (sliding window optimized)
  - Statistics: total return, annualized return, volatility, Sharpe, Calmar, max drawdown & duration
  - Performance: batch processing (`batch_size`), pre-extracted data, preallocated buffers, inlined hot paths

- Python API
  - Strategy lifecycle: `on_start` → `next(bar)` → `on_stop` with `on_order / on_trade` callbacks
  - Action format:
    - String: `"BUY" | "SELL"`
    - Dict: `{ "action": "BUY"|"SELL", "type": "market"|"limit", "size": float, "price"?: float }`
  - Data loader: CSV → list[dict] (MVP; pluggable for Parquet/Arrow)
  - Analyzers: drawdown segments, round-trips, enhanced performance metrics, factor backtests (quantiles/IC/monotonicity), unified report
  - **Cross-sectional Factor Analysis**: multi-factor evaluation system, quantile portfolio backtesting, IC/ICIR analysis, factor ranking
  - Optimizer: naive grid search (customizable scoring key)

- API & Frontend
  - FastAPI: `POST /runs`, `GET /runs`, `GET /runs/{id}`
  - Streamlit: submit runs, list & visualize results (equity curve + stats)

## Install & Build
Prereqs: Python 3.8+, Rust (`rustup`), maturin

```powershell
pip install maturin
cd rust/engine_rust
maturin develop --release
```

## Quick Start
- Minimal backtest
  ```powershell
  cd ../..
  python examples/run_mvp.py
  ```
- Analyzer demo
  ```powershell
  python examples/run_analyzers.py
  ```
- Grid search
  ```powershell
  python examples/run_grid_search.py
  ```
- Cross-sectional factor backtesting
  ```powershell
  python examples/run_cs_momentum_sample.py
  ```
- Quantile portfolio backtesting with trading simulation
  ```powershell
  python examples/run_cs_quantile_portfolios.py
  ```
- Performance test & batch-size comparison
  ```powershell
  python examples/run_performance_test.py
  ```

Sample data: `examples/data/sample.csv` (headers: `datetime,open,high,low,close,volume`).

## In Code
- Config & engine
  ```python
  from pyrust_bt.api import BacktestEngine, BacktestConfig
  cfg = BacktestConfig(start="2020-01-01", end="2020-12-31", cash=100000,
                       commission_rate=0.0005, slippage_bps=2.0, batch_size=1000)
  engine = BacktestEngine(cfg)
  ```
- Strategy (minimal)
  ```python
  from pyrust_bt.strategy import Strategy
  class MyStrategy(Strategy):
      def next(self, bar):
          if bar["close"] > 100:
              return {"action": "BUY", "type": "market", "size": 1.0}
          return None
  ```
- Run
  ```python
  from pyrust_bt.data import load_csv_to_bars
  bars = load_csv_to_bars("examples/data/sample.csv", symbol="SAMPLE")
  result = engine.run(MyStrategy(), bars)
  print(result["stats"], result["equity"])  # stats & equity
  ```

## Analysis & Reports
- Drawdown segments: `compute_drawdown_segments(equity_curve)`
- Round trips: `round_trips_from_trades(trades, bars)` / export CSV
- Performance metrics: `compute_performance_metrics(equity_curve)` (Sharpe/Sortino/Calmar/VAR)
- Factor backtest: `factor_backtest(bars, factor_key, quantiles, forward)`
- Unified report: `generate_analysis_report(...)`
- **Cross-sectional Factor Evaluation**:
  - Multi-factor analyzer: `MultiFactorAnalyzer` with time-series/cross-sectional methods
  - Factor ranking: IC, ICIR, monotonicity, stability, turnover analysis
  - Quantile portfolio backtesting: `CrossSectionFactorBacktester` for large-scale factor evaluation
  - Export: detailed reports, factor rankings, correlation matrices

## API & Frontend
- Start API (FastAPI)
  ```powershell
  pip install fastapi uvicorn pydantic requests streamlit
  python -m uvicorn python.server_main:app --reload
  ```
- Start frontend (Streamlit)
  ```powershell
  set PYRUST_BT_API=http://127.0.0.1:8000
  streamlit run frontend/streamlit_app.py
  ```

## Performance Notes
- Prefer larger `batch_size` (e.g., 1000–5000) to reduce Python round-trips
- Prefer dict actions over strings
- Use Rust vectorized indicators (`compute_sma/compute_rsi`) when possible
- For large data, prefer Parquet/Arrow and partitioned reads (by symbol/time)

## Project Structure
- `rust/engine_rust`: Rust engine (PyO3), indicators & stats
- `python/pyrust_bt`: Python API/strategy/data/analyzers/optimizer
- `python/pyrust_bt/multi_factor_analyzer.py`: Multi-factor evaluation system
- `python/pyrust_bt/cs_factor_backtester.py`: Cross-sectional factor backtesting
- `examples`: MVP, analyzers, grid search, performance tests
- `examples/run_cs_momentum_sample.py`: Cross-sectional momentum factor demo
- `examples/run_cs_quantile_portfolios.py`: Quantile portfolio trading simulation
- `frontend`: Streamlit UI

## TODO / Roadmap
- Engine/Matching
  - Partial fills, order book, stop/take-profit, OCO, conditional orders
  - Multi-asset/multi-timeframe alignment, calendar/timezone
  - Liquidity/impact models
- Data
  - Parquet/Arrow zero-copy pipelines, columnar batching
  - DataFeed abstraction (DB/object storage) & caching
- Analysis/Reports
  - Rich analyzers (group stats, drawdown visualization, trade distributions)
  - Report export (PDF/HTML) & multi-run comparison
  - Advanced factor analysis (industry/market cap neutralization, rolling quantiles)
- Optimization/Parallelism
  - Random/Bayesian search, cross-validation
  - Multi-process/distributed runs (Ray/Celery/k8s Jobs)
- Frontend/UX
  - React + ECharts/Plotly (task mgmt, playback, filters, annotations)
  - WebSocket live logs/progress/equity
- Quality/Eng
  - Unit/integration/regression tests, benchmarks
  - CI (wheel build/artifacts), releases

## 🚀 Performance Highlights
- Backtest speed: from 1,682 bars/s to **419,552 bars/s** (≈250×)
- Dataset: 550k bars in ~1.3s
- Memory: preallocated buffers to reduce reallocations
- Batching: configurable `batch_size` to reduce GIL contention

## Community
Pull requests are welcome!

![Join](images/yzbjs1.png)

## License
MIT

## Disclaimer
This tool is for research and education only and does not constitute investment advice. You are solely responsible for your trading decisions and associated risks. 