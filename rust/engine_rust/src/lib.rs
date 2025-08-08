use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// 预提取的bar数据结构
#[derive(Clone, Debug)]
struct BarData {
    datetime: Option<String>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[pyclass]
#[derive(Clone)]
pub struct BacktestConfig {
    #[pyo3(get)]
    pub start: String,
    #[pyo3(get)]
    pub end: String,
    #[pyo3(get)]
    pub cash: f64,
    #[pyo3(get)]
    pub commission_rate: f64,
    #[pyo3(get)]
    pub slippage_bps: f64,
    #[pyo3(get)]
    pub batch_size: usize,  // 新增：批处理大小
}

#[pymethods]
impl BacktestConfig {
    #[new]
    #[pyo3(signature = (start, end, cash, commission_rate=0.0, slippage_bps=0.0, batch_size=1000))]
    fn new(start: String, end: String, cash: f64, commission_rate: f64, slippage_bps: f64, batch_size: usize) -> Self {
        Self {
            start,
            end,
            cash,
            commission_rate,
            slippage_bps,
            batch_size,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum OrderSide {
    Buy,
    Sell,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum OrderType {
    Market,
    Limit,
}

#[derive(Clone, Debug)]
struct Order {
    id: u64,
    side: OrderSide,
    otype: OrderType,
    size: f64,
    limit_price: Option<f64>,
    status: &'static str,
}

#[derive(Default, Clone, Debug)]
struct PositionState {
    position: f64,
    avg_cost: f64,
    cash: f64,
    realized_pnl: f64,
}

impl PositionState {
    fn new(cash: f64) -> Self {
        Self {
            position: 0.0,
            avg_cost: 0.0,
            cash,
            realized_pnl: 0.0,
        }
    }
}

// 向量化指标计算（优化版）
pub fn vectorized_sma(prices: &[f64], window: usize) -> Vec<Option<f64>> {
    if prices.is_empty() || window == 0 {
        return vec![None; prices.len()];
    }
    
    let mut result = Vec::with_capacity(prices.len());
    let mut sum = 0.0;
    
    for i in 0..prices.len() {
        if i < window {
            sum += prices[i];
            result.push(None);
        } else if i == window {
            sum += prices[i];
            result.push(Some(sum / window as f64));
        } else {
            // 滑动窗口：减去最旧的，加上最新的
            sum = sum - prices[i - window] + prices[i];
            result.push(Some(sum / window as f64));
        }
    }
    result
}

pub fn vectorized_rsi(prices: &[f64], window: usize) -> Vec<Option<f64>> {
    if prices.len() < 2 || window == 0 {
        return vec![None; prices.len()];
    }
    
    let mut result = Vec::with_capacity(prices.len());
    result.push(None); // 第一个价格没有变化
    
    let mut gains = Vec::with_capacity(prices.len());
    let mut losses = Vec::with_capacity(prices.len());
    
    // 计算价格变化
    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // 计算RSI
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    
    for i in 0..gains.len() {
        if i < window - 1 {
            result.push(None);
        } else if i == window - 1 {
            // 初始平均
            avg_gain = gains[0..window].iter().sum::<f64>() / window as f64;
            avg_loss = losses[0..window].iter().sum::<f64>() / window as f64;
            
            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            };
            result.push(Some(rsi));
        } else {
            // Wilder的平滑方法
            avg_gain = ((avg_gain * (window - 1) as f64) + gains[i]) / window as f64;
            avg_loss = ((avg_loss * (window - 1) as f64) + losses[i]) / window as f64;
            
            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            };
            result.push(Some(rsi));
        }
    }
    
    result
}

#[pyfunction]
fn compute_sma(prices: Vec<f64>, window: usize) -> Vec<Option<f64>> {
    vectorized_sma(&prices, window)
}

#[pyfunction]
fn compute_rsi(prices: Vec<f64>, window: usize) -> Vec<Option<f64>> {
    vectorized_rsi(&prices, window)
}

// 批量提取bar数据，减少Python调用
fn extract_bars_data(bars: &PyList) -> PyResult<Vec<BarData>> {
    let mut bars_data = Vec::with_capacity(bars.len());
    
    for item in bars.iter() {
        let bar: &PyDict = item.downcast()?;
        
        let datetime = match bar.get_item("datetime")? {
            Some(v) => v.extract::<String>().ok(),
            None => None,
        };
        
        let open = bar.get_item("open")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0);
        let high = bar.get_item("high")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0);
        let low = bar.get_item("low")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0);
        let close = bar.get_item("close")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0);
        let volume = bar.get_item("volume")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(0.0);
        
        bars_data.push(BarData {
            datetime,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    
    Ok(bars_data)
}

#[pyclass]
pub struct BacktestEngine {
    cfg: BacktestConfig,
}

#[pymethods]
impl BacktestEngine {
    #[new]
    fn new(cfg: BacktestConfig) -> Self {
        Self { cfg }
    }

    /// 高性能回测循环：预提取数据、批量处理、减少Python调用
    fn run<'py>(&self, py: Python<'py>, strategy: PyObject, data: &'py PyAny) -> PyResult<PyObject> {
        let bars: &PyList = data.downcast()?;
        let n_bars = bars.len();

        // 预提取所有bar数据到Rust结构中
        let bars_data = extract_bars_data(bars)?;
        
        let _ = strategy.call_method1(py, "on_start", (py.None(),));

        let mut pos = PositionState::new(self.cfg.cash);
        let mut order_seq: u64 = 1;

        // 预分配容量
        let mut equity_curve: Vec<(Option<String>, f64)> = Vec::with_capacity(n_bars);
        let mut trades: Vec<(u64, String, f64, f64)> = Vec::with_capacity(n_bars / 100);

        // 批量处理策略调用，减少Python GIL争用
        let batch_size = self.cfg.batch_size.min(n_bars);
        
        for chunk_start in (0..n_bars).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(n_bars);
            
            // 处理当前批次
            for i in chunk_start..chunk_end {
                let bar_data = &bars_data[i];
                let last_price = bar_data.close;

                // 重新构造PyDict给策略（只在需要时）
                let bar_dict = PyDict::new_bound(py);
                if let Some(ref dt) = bar_data.datetime {
                    bar_dict.set_item("datetime", dt)?;
                }
                bar_dict.set_item("open", bar_data.open)?;
                bar_dict.set_item("high", bar_data.high)?;
                bar_dict.set_item("low", bar_data.low)?;
                bar_dict.set_item("close", bar_data.close)?;
                bar_dict.set_item("volume", bar_data.volume)?;

                let action_obj = strategy.call_method1(py, "next", (bar_dict.as_any(),))?;

                // 快速订单处理
                if let Some(order) = self.parse_action_fast(action_obj.as_ref(py), &mut order_seq, last_price)? {
                    if let Some((fill_price, fill_size)) = self.try_match(&order, last_price) {
                        let slip = self.cfg.slippage_bps / 10_000.0;
                        let sign = match order.side { OrderSide::Buy => 1.0, OrderSide::Sell => -1.0 };
                        let exec_price = fill_price * (1.0 + sign * slip);
                        let commission = exec_price * fill_size * self.cfg.commission_rate;

                        // 快速持仓更新
                        self.update_position(&mut pos, &order, exec_price, fill_size, commission);
                        trades.push((order.id, match order.side { OrderSide::Buy => "BUY".to_string(), OrderSide::Sell => "SELL".to_string() }, exec_price, fill_size));
                    }
                }

                let equity = pos.cash + pos.position * last_price;
                equity_curve.push((bar_data.datetime.clone(), equity));
            }
        }

        let _ = strategy.call_method0(py, "on_stop");

        // 构建结果（优化版）
        self.build_result(py, pos, equity_curve, trades)
    }
}

impl BacktestEngine {
    // 优化的动作解析，减少类型检查
    fn parse_action_fast<'py>(
        &self,
        action_obj: &PyAny,
        order_seq: &mut u64,
        last_price: f64,
    ) -> PyResult<Option<Order>> {
        // 快速字符串检查
        if let Ok(s) = action_obj.extract::<Option<String>>() {
            if let Some(act) = s {
                let side = if act.as_bytes()[0] == b'B' { OrderSide::Buy } else { OrderSide::Sell };
                let id = *order_seq; *order_seq += 1;
                return Ok(Some(Order { id, side, otype: OrderType::Market, size: 1.0, limit_price: None, status: "submitted" }));
            }
        }

        // 字典解析
        if let Ok(d) = action_obj.downcast::<PyDict>() {
            let act = d.get_item("action")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_default();
            if act.is_empty() { return Ok(None); }
            
            let side = if act.as_bytes()[0] == b'B' { OrderSide::Buy } else { OrderSide::Sell };
            let otype_str = d.get_item("type")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "market".into());
            let otype = if otype_str == "limit" { OrderType::Limit } else { OrderType::Market };
            let size = d.get_item("size")?.and_then(|v| v.extract::<f64>().ok()).unwrap_or(1.0);
            let price = d.get_item("price")?.and_then(|v| v.extract::<f64>().ok());
            
            let id = *order_seq; *order_seq += 1;
            let limit_price = if otype == OrderType::Limit { price.or(Some(last_price)) } else { None };
            return Ok(Some(Order { id, side, otype, size, limit_price, status: "submitted" }));
        }

        Ok(None)
    }

    #[inline]
    fn try_match(&self, order: &Order, last_price: f64) -> Option<(f64, f64)> {
        match order.otype {
            OrderType::Market => Some((last_price, order.size)),
            OrderType::Limit => {
                let lp = order.limit_price.unwrap_or(last_price);
                match order.side {
                    OrderSide::Buy => if last_price <= lp { Some((lp, order.size)) } else { None },
                    OrderSide::Sell => if last_price >= lp { Some((lp, order.size)) } else { None },
                }
            }
        }
    }

    #[inline]
    fn update_position(&self, pos: &mut PositionState, order: &Order, exec_price: f64, fill_size: f64, commission: f64) {
        match order.side {
            OrderSide::Buy => {
                let cost = exec_price * fill_size + commission;
                let new_pos = pos.position + fill_size;
                if new_pos.abs() > f64::EPSILON {
                    pos.avg_cost = if pos.position.abs() > f64::EPSILON {
                        (pos.avg_cost * pos.position + exec_price * fill_size) / new_pos
                    } else {
                        exec_price
                    };
                } else {
                    pos.avg_cost = 0.0;
                }
                pos.position = new_pos;
                pos.cash -= cost;
            }
            OrderSide::Sell => {
                let proceeds = exec_price * fill_size - commission;
                if pos.position > 0.0 {
                    let closing = fill_size.min(pos.position);
                    pos.realized_pnl += (exec_price - pos.avg_cost) * closing;
                }
                pos.position -= fill_size;
                if pos.position.abs() < f64::EPSILON { pos.avg_cost = 0.0; }
                pos.cash += proceeds;
            }
        }
    }

    fn build_result<'py>(&self, py: Python<'py>, pos: PositionState, equity_curve: Vec<(Option<String>, f64)>, trades: Vec<(u64, String, f64, f64)>) -> PyResult<PyObject> {
        let result = PyDict::new_bound(py);
        result.set_item("cash", pos.cash)?;
        result.set_item("position", pos.position)?;
        result.set_item("avg_cost", pos.avg_cost)?;
        result.set_item("equity", pos.cash + pos.position * equity_curve.last().map_or(0.0, |(_, eq)| *eq))?;
        result.set_item("realized_pnl", pos.realized_pnl)?;

        // 高效构建净值曲线
        let eq_list = PyList::empty_bound(py);
        for (dt, eq) in &equity_curve {
            let row = PyDict::new_bound(py);
            if let Some(d) = dt { row.set_item("datetime", d)?; } else { row.set_item("datetime", py.None())?; }
            row.set_item("equity", eq)?;
            eq_list.append(row)?;
        }
        result.set_item("equity_curve", eq_list)?;

        // 高效构建交易列表
        let tr_list = PyList::empty_bound(py);
        for (oid, side, price, size) in &trades {
            let t = PyDict::new_bound(py);
            t.set_item("order_id", oid)?;
            t.set_item("side", side)?;
            t.set_item("price", price)?;
            t.set_item("size", size)?;
            tr_list.append(t)?;
        }
        result.set_item("trades", tr_list)?;

        // 增强的统计分析
        let stats = self.compute_enhanced_stats(py, &equity_curve, &trades)?;
        result.set_item("stats", stats)?;

        Ok(result.into())
    }

    fn compute_enhanced_stats<'py>(&self, py: Python<'py>, equity_curve: &[(Option<String>, f64)], trades: &[(u64, String, f64, f64)]) -> PyResult<PyObject> {
        if equity_curve.is_empty() {
            return Ok(PyDict::new_bound(py).into());
        }
        
        let start_equity = equity_curve.first().unwrap().1;
        let end_equity = equity_curve.last().unwrap().1;
        let total_return = if start_equity != 0.0 { (end_equity / start_equity) - 1.0 } else { 0.0 };

        // 向量化收益率计算
        let mut returns: Vec<f64> = Vec::with_capacity(equity_curve.len().saturating_sub(1));
        for i in 1..equity_curve.len() {
            let prev = equity_curve[i-1].1;
            let curr = equity_curve[i].1;
            if prev != 0.0 { returns.push((curr / prev) - 1.0); }
        }

        let mean_return = if returns.is_empty() { 0.0 } else { returns.iter().sum::<f64>() / returns.len() as f64 };
        let var = if returns.len() > 1 {
            let sum_sq_diff: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum();
            sum_sq_diff / (returns.len() - 1) as f64
        } else { 0.0 };
        let std = var.sqrt();
        let sharpe = if std > 0.0 { (mean_return * 252.0_f64.sqrt()) / std } else { 0.0 };

        // 高效最大回撤计算
        let mut peak = start_equity;
        let mut max_dd: f64 = 0.0;
        let mut dd_duration = 0;
        let mut max_dd_duration = 0;
        
        for &(_, eq) in equity_curve {
            if eq > peak {
                peak = eq;
                dd_duration = 0;
            } else {
                dd_duration += 1;
                let current_dd = 1.0 - eq / peak;
                if current_dd > max_dd {
                    max_dd = current_dd;
                }
                if dd_duration > max_dd_duration {
                    max_dd_duration = dd_duration;
                }
            }
        }

        // 交易统计
        let total_trades = trades.len();
        let (winning_trades, losing_trades, total_pnl) = {
            let mut win = 0;
            let mut lose = 0;
            let mut pnl = 0.0;
            
            for i in 0..trades.len() {
                let (_, side, price, size) = &trades[i];
                if i > 0 {
                    let prev_price = trades[i-1].2;
                    let profit = if side == "BUY" { (price - prev_price) * size } else { (prev_price - price) * size };
                    pnl += profit;
                    if profit > 0.0 { win += 1; } else if profit < 0.0 { lose += 1; }
                }
            }
            (win, lose, pnl)
        };

        let win_rate = if total_trades > 0 { winning_trades as f64 / total_trades as f64 } else { 0.0 };
        let calmar = if max_dd > 0.0 { (mean_return * 252.0) / max_dd } else { 0.0 };

        let stats = PyDict::new_bound(py);
        stats.set_item("start_equity", start_equity)?;
        stats.set_item("end_equity", end_equity)?;
        stats.set_item("total_return", total_return)?;
        stats.set_item("annualized_return", mean_return * 252.0)?;
        stats.set_item("volatility", std * (252.0_f64.sqrt()))?;
        stats.set_item("sharpe", sharpe)?;
        stats.set_item("calmar", calmar)?;
        stats.set_item("max_drawdown", max_dd)?;
        stats.set_item("max_dd_duration", max_dd_duration)?;
        stats.set_item("total_trades", total_trades)?;
        stats.set_item("winning_trades", winning_trades)?;
        stats.set_item("losing_trades", losing_trades)?;
        stats.set_item("win_rate", win_rate)?;
        stats.set_item("total_pnl", total_pnl)?;
        
        Ok(stats.into())
    }
}

#[pymodule]
fn engine_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BacktestConfig>()?;
    m.add_class::<BacktestEngine>()?;
    m.add_function(wrap_pyfunction!(compute_sma, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rsi, m)?)?;
    Ok(())
} 