"""Experiment Builder — generate sweep commands to run in terminal or Colab."""
from __future__ import annotations

import streamlit as st

st.title("🧪 Experiment Builder")
st.caption("Configure your sweep here — the page generates the exact commands to run in terminal or Colab.")

intervals = st.multiselect(
    "Intervals", ["5m", "15m", "30m", "60m", "1d"],
    default=["15m", "30m", "60m", "1d"],
)
strategies = st.multiselect(
    "Strategies",
    ["ema_trend", "supertrend", "rsi_reversion", "donchian", "orb", "vwap"],
    default=["ema_trend", "supertrend", "rsi_reversion", "donchian"],
)
folds = st.slider("Walk-forward folds", 2, 8, 5)
max_combos = st.slider("Max param combos per cell", 10, 200, 60, 10)
objective = st.selectbox("Optimization objective", ["calmar", "sortino", "sharpe", "profit_factor"])
slippage = st.slider("Slippage (ticks)", 0.0, 5.0, 1.0, 0.5)
risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25)
top_n = st.slider("Ensemble top-N", 2, 8, 4)

if not intervals or not strategies:
    st.warning("Select at least one interval and one strategy to generate commands.")
    st.stop()

intervals_str = ",".join(intervals)
strategies_str = ",".join(strategies)
n_cells = len(intervals) * len(strategies)
est_mins_local = max(1, n_cells * folds * max_combos // 120)
est_mins_colab = max(1, est_mins_local // 3)

sweep_cmd = (
    f"python scripts/run_sweep.py \\\n"
    f"  --intervals {intervals_str} \\\n"
    f"  --strategies {strategies_str} \\\n"
    f"  --folds {folds} --max-combos {max_combos} \\\n"
    f"  --objective {objective} --slippage {slippage} --risk-pct {risk_pct} \\\n"
    f"  --out outputs/sweep_results.json"
)

ensemble_cmd = (
    f"python scripts/build_ensemble.py \\\n"
    f"  --top {top_n} --objective {objective} \\\n"
    f"  --sweep outputs/sweep_results.json \\\n"
    f"  --out outputs/ensemble_result.json"
)

# Colab config block that mirrors the sliders
colab_config = (
    f"INTERVALS      = '{intervals_str}'\n"
    f"STRATEGIES     = '{strategies_str}'\n"
    f"FOLDS          = {folds}\n"
    f"MAX_COMBOS     = {max_combos}\n"
    f"OBJECTIVE      = '{objective}'\n"
    f"SLIPPAGE       = {slippage}\n"
    f"RISK_PCT       = {risk_pct}\n"
    f"ENSEMBLE_TOP_N = {top_n}"
)

st.markdown("### Commands to run")
st.caption(
    f"~{n_cells} cells × {folds} folds × up to {max_combos} combos. "
    f"Est. runtime: **{est_mins_local}–{est_mins_local * 3} min locally**, "
    f"**{est_mins_colab}–{est_mins_colab * 3} min on Colab**."
)

# ── Option A: Terminal ─────────────────────────────────────────────────────────
with st.expander("Option A — Run locally (terminal)", expanded=True):
    st.markdown("**Step 1 — data (if not already done):**")
    st.code(
        "make import-1m   # import raw 1m CSV (~2 min)\n"
        "make resample    # build 3m/5m/15m/30m/60m intervals",
        language="bash",
    )
    st.markdown("**Step 2 — sweep:**")
    st.code(sweep_cmd, language="bash")
    st.markdown("**Step 3 — ensemble (after sweep finishes):**")
    st.code(ensemble_cmd, language="bash")

# ── Option B: Colab ───────────────────────────────────────────────────────────
with st.expander("Option B — Run on Google Colab (recommended for full matrix)", expanded=True):
    st.markdown(
        "**Open `notebooks/colab_sweep.ipynb`** in Colab and paste the config block below "
        "into the **Step 2 — Configuration** cell, then run all cells top-to-bottom."
    )

    st.markdown("**Paste into the config cell:**")
    st.code(colab_config, language="python")

    st.markdown("**Before opening Colab, upload one file to your Google Drive:**")
    st.code(
        "# Upload the raw 1m CSV to your Drive:\n"
        "# bank-nifty-1m-data.csv  →  My Drive/banknifty_bot/\n\n"
        "# The repo is cloned automatically from:\n"
        "# https://github.com/aditya33agrawal/banknifty-trading-bot",
        language="bash",
    )

    st.info(
        "After Colab finishes, the notebook **downloads** `sweep_results.json` and "
        "`ensemble_result.json` to your browser automatically. "
        "Then go to **🏆 Results Leaderboard → Upload from Colab / Drive** to load them."
    )

st.markdown("---")
st.info(
    "After the sweep runs (locally or on Colab), open the **🏆 Results Leaderboard** "
    "page in the sidebar to see ranked results and the Calmar heatmap."
)
