"""Results Leaderboard — ranked interval × strategy × param results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
SWEEP_PATH = REPO / "outputs" / "sweep_results.json"
ENSEMBLE_PATH = REPO / "outputs" / "ensemble_result.json"

st.title("🏆 Results Leaderboard")

# ── Source selector ────────────────────────────────────────────────────────────
with st.expander("📂 Load results", expanded=not SWEEP_PATH.exists()):
    src_tab, upload_tab = st.tabs(["Local file", "Upload from Colab / Drive"])

    with src_tab:
        if SWEEP_PATH.exists():
            st.success(f"Found `outputs/sweep_results.json` ({SWEEP_PATH.stat().st_size // 1024} KB)")
        else:
            st.warning("No local sweep results yet. Run `make sweep` or upload from Colab.")
        col_reload, col_dl = st.columns(2)
        if col_reload.button("🔄 Reload from disk"):
            st.cache_data.clear()
            st.rerun()
        if SWEEP_PATH.exists() and col_dl.download_button(
            "⬇ Download sweep_results.json",
            data=SWEEP_PATH.read_bytes(),
            file_name="sweep_results.json",
            mime="application/json",
        ):
            pass

    with upload_tab:
        st.markdown(
            "After running the Colab notebook, **download** `sweep_results.json` from Colab "
            "or your Drive `outputs/` folder, then upload it here."
        )
        uploaded_sweep = st.file_uploader(
            "sweep_results.json", type=["json"], key="sweep_upload",
            help="Upload the sweep_results.json file from Colab or Drive",
        )
        uploaded_ens = st.file_uploader(
            "ensemble_result.json (optional)", type=["json"], key="ens_upload",
        )
        if uploaded_sweep and st.button("💾 Save to outputs/ and reload"):
            SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
            SWEEP_PATH.write_bytes(uploaded_sweep.read())
            if uploaded_ens:
                ENSEMBLE_PATH.write_bytes(uploaded_ens.read())
            st.cache_data.clear()
            st.success("Saved. Reloading…")
            st.rerun()


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def _load_sweep_disk() -> list[dict]:
    if not SWEEP_PATH.exists():
        return []
    return json.loads(SWEEP_PATH.read_text())


# Prefer in-session upload if present
_raw: list[dict] = []
if "sweep_upload" in st.session_state and st.session_state.sweep_upload is not None:
    _upload_file = st.session_state.sweep_upload
    _upload_file.seek(0)
    _raw = json.load(_upload_file)
else:
    _raw = _load_sweep_disk()

df = pd.DataFrame(_raw) if _raw else pd.DataFrame()

col1, col2 = st.columns([3, 1])
col1.caption("Sorted by OOS Calmar. Gates = promotion criteria passed out of 5.")

# ── Main table ─────────────────────────────────────────────────────────────────
if df.empty:
    st.info(
        "No sweep results yet.\n\n"
        "**Option A — run locally:** `make sweep` in your terminal.\n\n"
        "**Option B — run on Colab:** open `notebooks/colab_sweep.ipynb`, run all cells, "
        "then upload the result JSON using the 📂 panel above."
    )
else:
    sort_col = st.selectbox(
        "Sort by",
        ["oos_calmar", "oos_sharpe", "oos_cagr_pct", "oos_profit_factor", "mc_prob_profit"],
        index=0,
    )
    df_sorted = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    display_cols = [
        "interval", "strategy", "oos_cagr_pct", "oos_sharpe", "oos_calmar",
        "oos_max_dd_pct", "oos_profit_factor", "oos_n_trades", "gates_passed", "mc_prob_profit",
    ]
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    st.dataframe(df_sorted[display_cols], use_container_width=True, hide_index=True)

    # Calmar heatmap
    if {"interval", "strategy", "oos_calmar"}.issubset(df.columns):
        st.markdown("### Calmar heatmap (interval × strategy)")
        pivot = df.pivot_table(index="strategy", columns="interval", values="oos_calmar", aggfunc="max")
        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", text_auto=".2f", aspect="auto")
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Drilldown
    st.markdown("### Drilldown")
    d1, d2 = st.columns(2)
    sel_interval = d1.selectbox("Interval", sorted(df["interval"].unique()))
    sel_strategy = d2.selectbox("Strategy", sorted(df["strategy"].unique()))
    row = df[(df["interval"] == sel_interval) & (df["strategy"] == sel_strategy)]
    if not row.empty:
        r = row.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("OOS CAGR", f"{r.get('oos_cagr_pct', 0):.1f}%")
        c2.metric("OOS Sharpe", f"{r.get('oos_sharpe', 0):.2f}")
        c3.metric("OOS Calmar", f"{r.get('oos_calmar', 0):.2f}")
        c4.metric("OOS Max DD", f"{r.get('oos_max_dd_pct', 0):.1f}%")
        c5.metric("Gates", f"{r.get('gates_passed', 0)}/5")
        st.markdown(f"**Best params:** `{r.get('best_params', '—')}`")
        mc_prob = r.get("mc_prob_profit", 0)
        mc_p5 = r.get("mc_p5_return_pct", 0)
        st.markdown(f"MC prob profit: `{mc_prob:.0%}` | p5 return: `{mc_p5:.1f}%`")
        st.caption(
            f"Data: {r.get('data_start', '?')} → {r.get('data_end', '?')} "
            f"({r.get('data_bars', 0):,} bars)"
        )

# ── Ensemble panel ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Ensemble result")

# Use uploaded ensemble if present, else disk
_ens_data: dict = {}
if "ens_upload" in st.session_state and st.session_state.ens_upload is not None:
    _ens_file = st.session_state.ens_upload
    _ens_file.seek(0)
    _ens_data = json.load(_ens_file)
elif ENSEMBLE_PATH.exists():
    _ens_data = json.loads(ENSEMBLE_PATH.read_text())

if _ens_data:
    m = _ens_data.get("blended_metrics", {})
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blended CAGR", f"{m.get('cagr_pct', 0):.1f}%")
        c2.metric("Blended Sharpe", f"{m.get('sharpe', 0):.2f}")
        c3.metric("Blended Calmar", f"{m.get('calmar', 0):.2f}")
        c4.metric("Blended MaxDD", f"{m.get('max_drawdown_pct', 0):.1f}%")

    members = _ens_data.get("members", [])
    if members:
        st.markdown("**Members:**")
        mem_display_cols = [
            "interval", "strategy", "member_weight",
            "backtest_cagr_pct", "backtest_sharpe", "backtest_calmar", "backtest_n_trades",
        ]
        mem_df = pd.DataFrame(members)
        mem_display_cols = [c for c in mem_display_cols if c in mem_df.columns]
        st.dataframe(mem_df[mem_display_cols], use_container_width=True, hide_index=True)

    corr = _ens_data.get("correlation_matrix")
    if corr:
        corr_df = pd.DataFrame(corr)
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values, x=list(corr_df.columns), y=list(corr_df.index),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row_] for row_ in corr_df.values],
            texttemplate="%{text}",
        ))
        fig.update_layout(title="Return correlation (lower = more diversification)",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Generated at: {_ens_data.get('generated_at', '?')}")
else:
    st.info("No ensemble result yet. Run `make ensemble` after the sweep completes.")
