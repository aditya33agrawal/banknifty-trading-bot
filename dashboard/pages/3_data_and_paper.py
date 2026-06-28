"""Data & Paper — data coverage and paper trading status."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]

st.title("🗂️ Data & Paper")

# Resolve config — prefer config_search.yaml, fall back to config_intraday.yaml
_search_cfg = REPO / "config" / "config_search.yaml"
_intraday_cfg = REPO / "config" / "config_intraday.yaml"
_cfg_path = _search_cfg if _search_cfg.exists() else _intraday_cfg

try:
    from banknifty_bot.config import load_config
    cfg = load_config(_cfg_path)
    processed_dir = cfg.data.processed_dir
    output_dir = cfg.run.output_dir
except Exception:
    processed_dir = "data/processed"
    output_dir = "outputs"

# ── Data coverage ──────────────────────────────────────────────────────────
st.markdown("## Data coverage")

manifest_path = REPO / processed_dir / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    for key, years in manifest.items():
        total = sum(y.get("rows", 0) for y in years.values())
        with st.expander(f"{key} — {total:,} rows across {len(years)} years"):
            st.dataframe(pd.DataFrame(years).T, use_container_width=True)
else:
    st.info(
        "No manifest yet. Import and resample data:\n\n"
        "```bash\nmake import-1m\nmake resample\n```"
    )

# ── Paper trading ──────────────────────────────────────────────────────────
st.markdown("## Paper trading")

state_path = REPO / output_dir / "paper_ensemble_state.json"
if not state_path.exists():
    st.info(
        "No paper state yet. Start with:\n\n"
        "`python scripts/paper_trade_ensemble.py --backfill 250`\n\n"
        "then run daily after the close:\n\n"
        "`python scripts/paper_trade_ensemble.py`"
    )
else:
    ps = json.loads(state_path.read_text())
    sleeves = ps["sleeves"]
    total_equity = sum(s["equity"] for s in sleeves.values())
    init = ps["initial_equity"]
    trades = pd.DataFrame(ps.get("trades", []))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Total equity",
        f"₹{total_equity:,.0f}",
        f"{(total_equity / init - 1) * 100:+.2f}%",
    )
    c2.metric("As of bar", str(ps.get("last_ts", "—"))[:10])
    c3.metric("Closed trades", f"{len(trades)}")

    sleeve_rows = [
        {
            "strategy": k,
            "equity": round(v["equity"], 0),
            "position": (
                f"{v['position']['side']} {v['position']['qty']} @ {v['position']['entry_price']:.0f}"
                if v.get("position")
                else "flat"
            ),
        }
        for k, v in sleeves.items()
    ]
    st.dataframe(pd.DataFrame(sleeve_rows), use_container_width=True, hide_index=True)

    if not trades.empty:
        st.markdown("**Paper trade ledger** (most recent first)")
        st.dataframe(trades.iloc[::-1], use_container_width=True, height=300)
        st.download_button(
            "⬇️ Download paper trades CSV",
            trades.to_csv(index=False),
            file_name="paper_ensemble_trades.csv",
            mime="text/csv",
        )

    st.caption(
        "Run `python scripts/paper_trade_ensemble.py` once per trading day after the close."
    )
