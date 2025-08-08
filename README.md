# pyrust-bt

混合语言回测框架：Python 负责策略与数据，Rust 负责高性能回测核心，通过 PyO3 无缝绑定。兼顾研发效率与执行性能，适合研究到生产的小团队落地。

## 功能概览
- 核心引擎（Rust）
  - 时间推进：按 bar/tick 顺序执行
  - 订单撮合：市价 / 限价（同 bar 简化成交）
  - 成本模型：手续费 `commission_rate`、滑点 `slippage_bps`
  - 仓位与账本：`position / avg_cost / cash / equity / realized_pnl`
  - 指标计算：向量化 `SMA / RSI`（滑动窗口优化）
  - 统计指标：总收益、年化、波动率、夏普、Calmar、最大回撤与持续时间
  - 性能优化：批处理（可配 `batch_size`）、预提取数据、预分配容器、内联热点函数

- Python API
  - 策略编程模型：`on_start` → `next(bar)` → `on_stop`，支持事件回调 `on_order / on_trade`
  - 动作格式：
    - 字符串：`"BUY" | "SELL"`
    - 结构化：`{"action": "BUY"|"SELL", "type": "market"|"limit", "size": float, "price"?: float}`
  - 数据加载：`CSV → list[dict]`（最简 MVP，可替换为 Parquet/Arrow）
  - 分析器：回撤段落、回合交易、增强性能指标、因子回测（分位/IC/单调性）、综合报告
  - 参数优化：朴素网格搜索（可自定义打分键）

- 前后端
  - API（FastAPI）：`POST /runs` 创建回测，`GET /runs`/`/runs/{id}` 查询
  - 前端（Streamlit）：提交任务、列表与结果展示、净值曲线与统计

## 安装与构建
前置：Python 3.8+、Rust（`rustup`）、maturin

```powershell
pip install maturin
cd rust/engine_rust
maturin develop --release
```

## 快速开始
- 基础回测
  ```powershell
  cd ../..
  python examples/run_mvp.py
  ```
- 分析器示例
  ```powershell
  python examples/run_analyzers.py
  ```
- 网格搜索示例
  ```powershell
  python examples/run_grid_search.py
  ```
- 性能测试与批处理对比
  ```powershell
  python examples/run_performance_test.py
  ```

示例数据默认读取 `examples/data/sample.csv`（可替换为自己的 CSV：headers: `datetime,open,high,low,close,volume`）。

## 在代码中使用
- 配置与引擎
  ```python
  from pyrust_bt.api import BacktestEngine, BacktestConfig
  cfg = BacktestConfig(start="2020-01-01", end="2020-12-31", cash=100000,
                       commission_rate=0.0005, slippage_bps=2.0, batch_size=1000)
  engine = BacktestEngine(cfg)
  ```
- 策略接口（最小实现）
  ```python
  from pyrust_bt.strategy import Strategy
  class MyStrategy(Strategy):
      def next(self, bar):
          if bar["close"] > 100:
              return {"action": "BUY", "type": "market", "size": 1.0}
          return None
  ```
- 运行回测
  ```python
  from pyrust_bt.data import load_csv_to_bars
  bars = load_csv_to_bars("examples/data/sample.csv", symbol="SAMPLE")
  result = engine.run(MyStrategy(), bars)
  print(result["stats"], result["equity"])  # 统计指标与当前净值
  ```

## 分析与报告
- 回撤段落：`compute_drawdown_segments(equity_curve)`
- 回合交易：`round_trips_from_trades(trades, bars)` / 导出 CSV
- 性能指标：`compute_performance_metrics(equity_curve)`（Sharpe/Sortino/Calmar/VAR等）
- 因子回测：`factor_backtest(bars, factor_key, quantiles, forward)`
- 综合报告：`generate_analysis_report(equity_curve, trades, round_trips, drawdown_segments)`

## API 服务与前端
- 启动 API（FastAPI）
  ```powershell
  pip install fastapi uvicorn pydantic requests streamlit
  python -m uvicorn python.server_main:app --reload
  ```
- 启动前端（Streamlit）
  ```powershell
  set PYRUST_BT_API=http://127.0.0.1:8000
  streamlit run frontend/streamlit_app.py
  ```

## 性能提示
- 使用较大的 `batch_size`（如 1000~5000）以减少 Python 往返
- 尽量采用结构化动作（dict），避免多余字符串处理
- 指标尽量用 Rust 侧向量化函数（`compute_sma/compute_rsi`）
- 大数据优先 Parquet/Arrow，并考虑分区读（按 symbol/time）

## 🚀 性能优化成果
- **回测速度**: 从 1,682 bars/s 提升至 **419,552 bars/s**（提升 250倍）
- **55万根bar数据**: 回测耗时仅 1.3秒
- **内存优化**: 预分配容器，减少重分配开销
- **批处理**: 可配置批处理大小，减少Python GIL争用

## 目录结构
- `rust/engine_rust`：Rust 回测引擎（PyO3 扩展，含指标与统计）
- `python/pyrust_bt`：Python API/策略/数据/分析器/优化器
- `examples`：示例脚本（MVP、分析器、网格搜索、性能测试）
- `frontend`：Streamlit 前端

## TODO / Roadmap
- 引擎/撮合
  - 部分成交、挂单簿、止损/止盈、OCO、条件单
  - 多资产/多周期对齐（联合推进）、时区与交易日历
  - 更细的成交/流动性/冲击模型
- 数据
  - Parquet/Arrow 零拷贝管道、列式批处理
  - DataFeed 抽象（数据库/对象存储接入）与缓存策略
- 分析/报告
  - 更丰富的 Analyzer（分组统计、回撤段落可视化、交易分布）
  - 报告导出（PDF/HTML）与对比多次回测
- 优化/并行
  - 随机搜索/贝叶斯优化、交叉验证
  - 多进程/分布式参数搜索（Ray/Celery/k8s Jobs）
- 前端/可视化
  - React + ECharts/Plotly 的交互式 UI（任务管理、回放、筛选、标注）
  - WebSocket 实时日志/进度/曲线流
- 质量/工程化
  - 单元/集成/回归测试、基准测试
  - CI（构建 wheel、上传 artifacts）、发布

## 许可
MIT 

## 交流学习

欢迎提交PR！

![加群](images/yzbjs1.png)


## 免责声明

本工具提供的分析仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。用户应对自己的投资决策负责。

