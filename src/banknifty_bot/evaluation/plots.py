from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def equity_curve_fig(equity_df: pd.DataFrame, gross_equity_df: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df["equity"], name="Net equity"))
    if gross_equity_df is not None:
        fig.add_trace(go.Scatter(x=gross_equity_df.index, y=gross_equity_df["equity"], name="Gross equity"))
    fig.update_layout(title="Equity curve", xaxis_title="Time", yaxis_title="Equity")
    return fig


def drawdown_fig(equity_df: pd.DataFrame) -> go.Figure:
    equity = equity_df["equity"]
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown %"))
    fig.update_layout(title="Underwater curve", xaxis_title="Time", yaxis_title="Drawdown (%)")
    return fig


def monthly_returns_heatmap_fig(equity_df: pd.DataFrame) -> go.Figure:
    monthly = equity_df["equity"].resample("ME").last().pct_change().dropna() * 100
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn"))
    fig.update_layout(title="Monthly returns (%)", xaxis_title="Month", yaxis_title="Year")
    return fig


def trade_pnl_histogram_fig(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=trades_df["net_pnl"])])
    fig.update_layout(title="Trade P&L distribution", xaxis_title="Net P&L", yaxis_title="Count")
    return fig


def pnl_by_time_of_day_fig(trades_df: pd.DataFrame) -> go.Figure:
    df = trades_df.copy()
    df["entry_hour"] = pd.to_datetime(df["entry_time"]).dt.hour
    grouped = df.groupby("entry_hour")["net_pnl"].mean()
    fig = go.Figure(data=[go.Bar(x=grouped.index, y=grouped.values)])
    fig.update_layout(title="Avg P&L by entry hour", xaxis_title="Hour", yaxis_title="Avg net P&L")
    return fig


def pnl_by_weekday_fig(trades_df: pd.DataFrame) -> go.Figure:
    df = trades_df.copy()
    df["weekday"] = pd.to_datetime(df["entry_time"]).dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    grouped = df.groupby("weekday")["net_pnl"].mean().reindex(order)
    fig = go.Figure(data=[go.Bar(x=grouped.index, y=grouped.values)])
    fig.update_layout(title="Avg P&L by weekday", xaxis_title="Weekday", yaxis_title="Avg net P&L")
    return fig


def rolling_sharpe_fig(equity_df: pd.DataFrame, window: int = 20) -> go.Figure:
    daily = equity_df["equity"].resample("1D").last().dropna().pct_change().dropna()
    rolling_sharpe = (daily.rolling(window).mean() / daily.rolling(window).std()) * (252 ** 0.5)
    fig = go.Figure(data=[go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe)])
    fig.update_layout(title=f"Rolling {window}-day Sharpe", xaxis_title="Date", yaxis_title="Sharpe")
    return fig
