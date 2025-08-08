from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable, Optional
import csv
import math


@dataclass
class DrawdownSegment:
    start_idx: int
    peak_idx: int
    trough_idx: int
    end_idx: int
    peak_equity: float
    trough_equity: float
    drawdown: float  # as fraction, e.g. 0.12
    duration: int  # bars in drawdown
    recovery_time: Optional[int] = None  # bars to recover


def compute_drawdown_segments(equity_curve: List[Dict[str, Any]]) -> List[DrawdownSegment]:
    if not equity_curve:
        return []
    
    segments: List[DrawdownSegment] = []
    peak = equity_curve[0]["equity"]
    peak_idx = 0
    trough = peak
    trough_idx = 0
    in_dd = False
    
    for i, row in enumerate(equity_curve):
        eq = float(row["equity"])  # type: ignore[assignment]
        if eq > peak:
            if in_dd and peak > 0:
                duration = trough_idx - peak_idx + 1
                recovery_time = i - trough_idx if i > trough_idx else None
                segments.append(
                    DrawdownSegment(
                        start_idx=peak_idx,
                        peak_idx=peak_idx,
                        trough_idx=trough_idx,
                        end_idx=i - 1,
                        peak_equity=peak,
                        trough_equity=trough,
                        drawdown=1.0 - (trough / peak),
                        duration=duration,
                        recovery_time=recovery_time,
                    )
                )
            peak = eq
            peak_idx = i
            trough = eq
            trough_idx = i
            in_dd = False
        else:
            if eq < trough:
                trough = eq
                trough_idx = i
                in_dd = True
    
    if in_dd and peak > 0:
        last_idx = len(equity_curve) - 1
        duration = last_idx - peak_idx + 1
        segments.append(
            DrawdownSegment(
                start_idx=peak_idx,
                peak_idx=peak_idx,
                trough_idx=trough_idx,
                end_idx=last_idx,
                peak_equity=peak,
                trough_equity=trough,
                drawdown=1.0 - (trough / peak),
                duration=duration,
                recovery_time=None,  # Still in drawdown
            )
        )
    
    segments.sort(key=lambda s: s.drawdown, reverse=True)
    return segments


@dataclass
class RoundTrip:
    entry_idx: int
    exit_idx: int
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_ratio: float
    holding_bars: int
    entry_datetime: Optional[str] = None
    exit_datetime: Optional[str] = None


def round_trips_from_trades(
    trades: List[Dict[str, Any]],
    bars: Optional[List[Dict[str, Any]]] = None,
) -> List[RoundTrip]:
    rts: List[RoundTrip] = []
    inv_side: Optional[str] = None
    inv_price: float = 0.0
    inv_size: float = 0.0
    inv_idx: int = 0

    for i, t in enumerate(trades):
        side = str(t.get("side", "")).upper()
        price = float(t.get("price", 0.0))
        size = float(t.get("size", 0.0))
        
        if inv_side is None:
            inv_side = "LONG" if side == "BUY" else "SHORT"
            inv_price = price
            inv_size = size
            inv_idx = i
        else:
            if inv_side == "LONG" and side == "SELL":
                closing = min(inv_size, size)
                pnl = (price - inv_price) * closing
                ret = (price / inv_price - 1.0)
                
                entry_dt = bars[inv_idx].get("datetime") if bars and inv_idx < len(bars) else None
                exit_dt = bars[i].get("datetime") if bars and i < len(bars) else None
                
                rts.append(
                    RoundTrip(
                        entry_idx=inv_idx,
                        exit_idx=i,
                        side="LONG",
                        entry_price=inv_price,
                        exit_price=price,
                        size=closing,
                        pnl=pnl,
                        return_ratio=ret,
                        holding_bars=(i - inv_idx),
                        entry_datetime=entry_dt,
                        exit_datetime=exit_dt,
                    )
                )
                inv_side = None
            elif inv_side == "SHORT" and side == "BUY":
                closing = min(inv_size, size)
                pnl = (inv_price - price) * closing
                ret = (inv_price / price - 1.0)
                
                entry_dt = bars[inv_idx].get("datetime") if bars and inv_idx < len(bars) else None
                exit_dt = bars[i].get("datetime") if bars and i < len(bars) else None
                
                rts.append(
                    RoundTrip(
                        entry_idx=inv_idx,
                        exit_idx=i,
                        side="SHORT",
                        entry_price=inv_price,
                        exit_price=price,
                        size=closing,
                        pnl=pnl,
                        return_ratio=ret,
                        holding_bars=(i - inv_idx),
                        entry_datetime=entry_dt,
                        exit_datetime=exit_dt,
                    )
                )
                inv_side = None
    return rts


def export_trades_to_csv(trades: List[Dict[str, Any]], path: str) -> None:
    if not trades:
        return
    keys = list(trades[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trades)


def export_round_trips_to_csv(round_trips: List[RoundTrip], path: str) -> None:
    if not round_trips:
        return
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "entry_idx", "exit_idx", "side", "entry_price", "exit_price",
            "size", "pnl", "return_ratio", "holding_bars", "entry_datetime", "exit_datetime"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for rt in round_trips:
            writer.writerow({
                "entry_idx": rt.entry_idx,
                "exit_idx": rt.exit_idx,
                "side": rt.side,
                "entry_price": rt.entry_price,
                "exit_price": rt.exit_price,
                "size": rt.size,
                "pnl": rt.pnl,
                "return_ratio": rt.return_ratio,
                "holding_bars": rt.holding_bars,
                "entry_datetime": rt.entry_datetime,
                "exit_datetime": rt.exit_datetime,
            })


def compute_performance_metrics(equity_curve: List[Dict[str, Any]], risk_free_rate: float = 0.02) -> Dict[str, float]:
    """计算增强的性能指标"""
    if len(equity_curve) < 2:
        return {}
    
    # 提取净值序列
    values = [float(row["equity"]) for row in equity_curve]
    
    # 计算收益率
    returns = []
    for i in range(1, len(values)):
        if values[i-1] != 0:
            returns.append((values[i] / values[i-1]) - 1.0)
    
    if not returns:
        return {}
    
    # 基础统计
    total_return = (values[-1] / values[0]) - 1.0 if values[0] != 0 else 0.0
    mean_return = sum(returns) / len(returns)
    
    # 波动率计算
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
    volatility = math.sqrt(variance)
    
    # 年化指标（假设252个交易日）
    annualized_return = mean_return * 252
    annualized_volatility = volatility * math.sqrt(252)
    
    # 夏普比率
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
    
    # 最大回撤
    peak = values[0]
    max_dd = 0.0
    dd_duration = 0
    max_dd_duration = 0
    current_dd_duration = 0
    
    for value in values:
        if value > peak:
            peak = value
            current_dd_duration = 0
        else:
            current_dd_duration += 1
            dd = (peak - value) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if current_dd_duration > max_dd_duration:
                max_dd_duration = current_dd_duration
    
    # Calmar比率
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0
    
    # Sortino比率（下行风险）
    negative_returns = [r for r in returns if r < 0]
    downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns) if negative_returns else 0.0
    downside_deviation = math.sqrt(downside_variance) * math.sqrt(252)
    sortino = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
    
    # VaR (5%)
    sorted_returns = sorted(returns)
    var_5 = sorted_returns[int(len(sorted_returns) * 0.05)] if sorted_returns else 0.0
    
    # 胜率
    winning_periods = sum(1 for r in returns if r > 0)
    win_rate = winning_periods / len(returns) if returns else 0.0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_dd_duration": max_dd_duration,
        "var_5": var_5,
        "win_rate": win_rate,
    }


def factor_backtest(
    bars: List[Dict[str, Any]],
    factor_key: str,
    quantiles: int = 5,
    forward: int = 1,
) -> Dict[str, Any]:
    """增强的因子回测分析"""
    n = len(bars)
    if n <= forward or quantiles < 2:
        return {"quantiles": [], "mean_returns": [], "ic": None}

    # 计算前向收益
    fwd_returns: List[float] = []
    for i in range(n - forward):
        c0 = float(bars[i]["close"])  # type: ignore[assignment]
        c1 = float(bars[i + forward]["close"])  # type: ignore[assignment]
        fwd_returns.append((c1 / c0) - 1.0 if c0 != 0 else 0.0)

    # 收集因子值
    factor_values: List[float] = []
    for i in range(n - forward):
        v = bars[i].get(factor_key)
        factor_values.append(float(v) if v is not None else 0.0)

    if not factor_values or not fwd_returns:
        return {"quantiles": [], "mean_returns": [], "ic": None}

    # 分位数分组
    sorted_factors = sorted(factor_values)
    q_bounds = []
    for q in range(1, quantiles):
        idx = int(len(sorted_factors) * q / quantiles)
        q_bounds.append(sorted_factors[idx])

    # 分组统计
    groups: List[List[float]] = [[] for _ in range(quantiles)]
    for fac_val, ret in zip(factor_values, fwd_returns):
        group_idx = 0
        for bound in q_bounds:
            if fac_val > bound:
                group_idx += 1
            else:
                break
        groups[group_idx].append(ret)

    mean_returns = [sum(g) / len(g) if g else 0.0 for g in groups]
    
    # IC计算（皮尔逊相关）
    if len(factor_values) == len(fwd_returns) and len(factor_values) > 1:
        mean_fac = sum(factor_values) / len(factor_values)
        mean_ret = sum(fwd_returns) / len(fwd_returns)
        
        cov = sum((f - mean_fac) * (r - mean_ret) for f, r in zip(factor_values, fwd_returns))
        var_fac = sum((f - mean_fac) ** 2 for f in factor_values)
        var_ret = sum((r - mean_ret) ** 2 for r in fwd_returns)
        
        ic = cov / (math.sqrt(var_fac * var_ret) + 1e-12)
    else:
        ic = 0.0

    # 计算分组间的单调性
    monotonicity = 0.0
    if len(mean_returns) > 1:
        increasing = sum(1 for i in range(1, len(mean_returns)) if mean_returns[i] > mean_returns[i-1])
        decreasing = sum(1 for i in range(1, len(mean_returns)) if mean_returns[i] < mean_returns[i-1])
        monotonicity = (increasing - decreasing) / (len(mean_returns) - 1)

    return {
        "quantiles": list(range(1, quantiles + 1)),
        "mean_returns": mean_returns,
        "ic": ic,
        "monotonicity": monotonicity,
        "q_bounds": q_bounds,
        "factor_stats": {
            "mean": sum(factor_values) / len(factor_values) if factor_values else 0.0,
            "std": math.sqrt(sum((f - sum(factor_values) / len(factor_values)) ** 2 for f in factor_values) / len(factor_values)) if len(factor_values) > 1 else 0.0,
            "min": min(factor_values) if factor_values else 0.0,
            "max": max(factor_values) if factor_values else 0.0,
        }
    }


def generate_analysis_report(
    equity_curve: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    round_trips: List[RoundTrip],
    drawdown_segments: List[DrawdownSegment],
) -> Dict[str, Any]:
    """生成综合分析报告"""
    
    # 基础统计
    performance = compute_performance_metrics(equity_curve)
    
    # 交易统计
    total_trades = len(trades)
    profitable_rts = [rt for rt in round_trips if rt.pnl > 0]
    losing_rts = [rt for rt in round_trips if rt.pnl < 0]
    
    trade_stats = {
        "total_round_trips": len(round_trips),
        "profitable_trades": len(profitable_rts),
        "losing_trades": len(losing_rts),
        "win_rate": len(profitable_rts) / len(round_trips) if round_trips else 0.0,
        "avg_win": sum(rt.pnl for rt in profitable_rts) / len(profitable_rts) if profitable_rts else 0.0,
        "avg_loss": sum(rt.pnl for rt in losing_rts) / len(losing_rts) if losing_rts else 0.0,
        "profit_factor": sum(rt.pnl for rt in profitable_rts) / abs(sum(rt.pnl for rt in losing_rts)) if losing_rts and sum(rt.pnl for rt in losing_rts) != 0 else float('inf'),
        "avg_holding_period": sum(rt.holding_bars for rt in round_trips) / len(round_trips) if round_trips else 0.0,
    }
    
    # 回撤统计
    dd_stats = {
        "max_drawdown": max(seg.drawdown for seg in drawdown_segments) if drawdown_segments else 0.0,
        "avg_drawdown": sum(seg.drawdown for seg in drawdown_segments) / len(drawdown_segments) if drawdown_segments else 0.0,
        "max_dd_duration": max(seg.duration for seg in drawdown_segments) if drawdown_segments else 0,
        "avg_dd_duration": sum(seg.duration for seg in drawdown_segments) / len(drawdown_segments) if drawdown_segments else 0.0,
        "total_dd_periods": len(drawdown_segments),
    }
    
    return {
        "performance": performance,
        "trades": trade_stats,
        "drawdowns": dd_stats,
        "summary": {
            "start_value": equity_curve[0]["equity"] if equity_curve else 0.0,
            "end_value": equity_curve[-1]["equity"] if equity_curve else 0.0,
            "total_bars": len(equity_curve),
            "analysis_timestamp": None,  # 可以添加时间戳
        }
    } 