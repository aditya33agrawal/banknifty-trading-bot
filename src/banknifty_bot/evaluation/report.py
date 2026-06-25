from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics as M
from . import plots as P


def _metrics_table_html(metrics: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    return f"<table border='1' cellpadding='6'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"


def generate_html_report(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    run_meta: dict,
    output_path: str | Path,
) -> Path:
    """Single self-contained HTML report: metrics table + charts + run metadata
    (data coverage, cost assumptions, params) — the go/no-go artifact (plan §8).
    """
    report_metrics = M.full_report(equity_df, trades_df)

    figs_html = []
    figs_html.append(P.equity_curve_fig(equity_df).to_html(full_html=False, include_plotlyjs="cdn"))
    figs_html.append(P.drawdown_fig(equity_df).to_html(full_html=False, include_plotlyjs=False))
    if not trades_df.empty:
        figs_html.append(P.trade_pnl_histogram_fig(trades_df).to_html(full_html=False, include_plotlyjs=False))
        figs_html.append(P.pnl_by_time_of_day_fig(trades_df).to_html(full_html=False, include_plotlyjs=False))
        figs_html.append(P.pnl_by_weekday_fig(trades_df).to_html(full_html=False, include_plotlyjs=False))
    try:
        figs_html.append(P.monthly_returns_heatmap_fig(equity_df).to_html(full_html=False, include_plotlyjs=False))
    except Exception:
        pass

    meta_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in run_meta.items())

    html = f"""
    <html><head><title>Backtest report — {run_meta.get('strategy', '')}</title></head>
    <body style="font-family: sans-serif; max-width: 1100px; margin: auto;">
        <h1>Backtest Report</h1>
        <h2>Run metadata</h2>
        <table border='1' cellpadding='6'>{meta_rows}</table>
        <h2>Metrics</h2>
        {_metrics_table_html(report_metrics)}
        <h2>Charts</h2>
        {''.join(figs_html)}
    </body></html>
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
