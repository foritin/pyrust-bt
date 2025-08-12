# Changelog / 更新日志

## [Unreleased]
- 引擎/撮合：部分成交、挂单簿、止损/止盈、OCO、条件单
- 多资产结果：在回测结果中补充逐 symbol 的期末头寸与成本（`positions_by_symbol`）
- 时间与日历：交易日历/时区，对齐策略（多周期、跨市场）
- 数据层：Parquet/Arrow 零拷贝管道，DataFeed 抽象与缓存
- 分析器：分组统计、回撤可视化、交易分布图；多回测结果对比
- 优化与并行：随机/贝叶斯搜索、交叉验证；分布式参数搜索（Ray/Celery/k8s Jobs）
- 前端：React + ECharts/Plotly 交互式 UI；WebSocket 实时日志/进度/曲线
- 质量工程：完善单元/集成/回归测试，基准测试；CI 构建与发布

## [0.1.1] - 2025-08-12
### 新增
- 多资产/多周期回测（MVP）：
  - `BacktestEngine.run_multi(strategy, feeds)`，联合时间线推进，`feeds: Dict[str, List[bar]]`
  - 订单支持 `symbol` 字段；事件回调 `on_order/on_trade` 携带 `symbol`
  - 策略优先实现 `next_multi(update_slice, ctx)`，回退到 `next(bar, ctx)`
- 上下文 `EngineContext`：在 `on_start(ctx)` 与 `next(bar, ctx)` 提供 `position/avg_cost/cash/equity/bar_index`
- 事件回调：恢复并完善 `on_order`（submitted/filled）与 `on_trade`
- 因子回测加速：Rust 端 `factor_backtest_fast` + Python 自动分流（>5k bars）
- 新增示例：
  - `examples/run_multi_assets.py` + `examples/data/multi_assets.csv`
  - 性能测试 `examples/run_performance_test.py`（批处理对比）
  - 分析器 `examples/run_analyzers.py`
  - 网格搜索 `examples/run_grid_search.py`

### 变更
- `Strategy.next` 兼容两种签名：`next(bar, ctx)` 优先；无 `ctx` 时回退到 `next(bar)`
- `pyrust_bt.api.BacktestEngine` 新增 `run_multi`

### 注意（兼容性）
- 如策略实现了 `next(bar)` 但未实现 `next(bar, ctx)`，引擎会自动回退，无需改动即可运行
- 建议策略实现/使用 `on_order` 与 `on_trade` 以获取更丰富的执行信息


## [0.1.0] - 2025-08-08

### 初版（MVP）
- PyO3 绑定的 Rust 引擎 + Python 包装层
- 单资产回测：市价/限价、手续费/滑点、仓位与账本
- 分析器：基础收益与风险指标、回撤、回合交易、因子回测（Python 实现）
- 示例：`run_mvp.py`、`examples/data/sample.csv`
- 工程：`.gitignore`、`README.md`

### 性能
- 单资产回测循环优化：
  - 预提取数据（减少 Python↔Rust 往返）
  - 批处理 `batch_size`，降低 GIL 争用
  - 向量化指标 `SMA/RSI`（滑动窗口 O(1) 更新）
  - 预分配容器、内联关键路径
- 实测从 ~1,682 bars/s 提升至 ~419,552 bars/s（≈250×，55 万 bar ~1.3s）

