# pyrust-bt (高性能版)

Hybrid Python + Rust backtesting framework per PRD.

## 🚀 性能优化成果
- **回测速度**: 从 1,682 bars/s 提升至 **419,552 bars/s**（提升 250倍）
- **55万根bar数据**: 回测耗时仅 1.3秒
- **内存优化**: 预分配容器，减少重分配开销
- **批处理**: 可配置批处理大小，减少Python GIL争用

## 新增功能
- **性能优化**
  - 预提取bar数据到Rust结构，减少Python调用
  - 向量化指标计算（SMA、RSI）
  - 批量处理策略调用
  - 内联函数优化关键路径
  
- **增强分析器**
  - 回撤段落分析（持续时间、恢复时间）
  - 回合交易统计（入场/出场时间、持仓周期）
  - 增强性能指标（Sortino、Calmar、VaR等）
  - 因子回测（分位数分组、IC分析、单调性）
  - 综合分析报告生成

- **订单与成本模型**
  - 市价/限价单即时撮合
  - 手续费与滑点模拟
  - 持仓成本跟踪
  - 已实现/未实现盈亏

## 快速开始（Windows / PowerShell）
先构建扩展：
```powershell
cd rust/engine_rust
maturin develop --release
```

运行性能测试：
```powershell
cd ../..
python examples/run_performance_test.py
```

运行分析器示例：
```powershell
python examples/run_analyzers.py
```

运行网格搜索：
```powershell
python examples/run_grid_search.py
```

## API 服务与前端
- 启动 API（FastAPI）：
  ```powershell
  pip install fastapi uvicorn pydantic requests streamlit
  python -m uvicorn python.server_main:app --reload
  ```
- 启动前端（Streamlit）：
  ```powershell
  set PYRUST_BT_API=http://127.0.0.1:8000
  streamlit run frontend/streamlit_app.py
  ```

## 性能基准
| 数据量 | 回测时间 | 速度 | 内存使用 |
|--------|----------|------|----------|
| 55万根bar | 1.3s | 42万 bars/s | ~100MB |
| 分析计算 | 0.9s | - | ~50MB |

## 目录
- `rust/engine_rust`: 高性能 Rust 引擎（PyO3）
- `python/pyrust_bt`: Python API / Strategy / Data / Indicators / Optimize / Analyzers
- `examples`: 运行示例（基础、性能测试、网格搜索、分析器）
- `frontend`: Streamlit 可视化界面

## 架构特点
- **零拷贝**: 预提取数据到Rust，避免重复序列化
- **批处理**: 可配置批量大小，平衡性能与内存
- **向量化**: Rust实现的高效指标计算
- **内联优化**: 关键路径函数内联，减少调用开销
- **内存友好**: 预分配容器，避免频繁重分配

## 下一步
- WebAssembly支持（浏览器内回测）
- GPU加速指标计算
- 分布式并行回测
- 实时数据接入
- 更丰富的资产类别支持 