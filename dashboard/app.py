"""BankNifty Bot — dashboard router.

Run: streamlit run dashboard/app.py   (or: make dashboard)

Three pages (auto-discovered from dashboard/pages/):
  1. Experiment Builder  — generate sweep commands to run in terminal/Colab
  2. Results Leaderboard — view ranked interval × strategy × param results
  3. Data & Paper        — data coverage and paper trading status
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="BankNifty Bot", layout="wide", page_icon="📈")
st.title("📈 BankNifty Bot")
st.markdown("""
Welcome. Use the **sidebar** to navigate between pages.

---

### Workflow

| Step | What to do |
|------|------------|
| **1. Import data** | `make import-1m` — loads the 11-year 1m CSV into the store |
| **2. Resample** | `make resample` — builds 3m / 5m / 15m / 30m / 60m from 1m |
| **3. Build commands** | Open **🧪 Experiment Builder** in the sidebar, configure, copy commands |
| **4. Run sweep** | Paste and run in terminal or Colab — results land in `outputs/` |
| **5. View results** | Open **🏆 Results Leaderboard** — ranked by OOS Calmar |
| **6. Blend winners** | `make ensemble` — multi-timeframe ensemble from top-N cells |

---

### Quick commands
```bash
make import-1m    # import raw 1m data
make resample     # resample to all intervals
make sweep        # run the full interval × strategy matrix
make ensemble     # build ensemble from top-N results
make dashboard    # start this dashboard
```
""")
