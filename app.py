"""
app.py: Reliability in GENOME. A dashboard on how countries behave toward the parties of a conflict.

Run with:  .venv/bin/streamlit run app.py   (from the project root)

Data: free count queries stored in data/processed/ (built by scripts/build_all.sh) plus live free
page-1 event lists. No API credits are spent by this app.
"""

import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
for folder in (os.path.join(ROOT, "src"), ROOT):   # src: logic; ROOT: the `views` package
    if folder not in sys.path:
        sys.path.insert(0, folder)

import metrics as mt
from views import analysis, case_page, chains_page, clusters_page, map_page, placeholders
from views.common import load_data

st.set_page_config(page_title="Reliability in GENOME", page_icon="🌍", layout="wide")

if not mt.COUNTS.exists():
    st.error("No data yet. Run `zsh scripts/build_all.sh` first (free, see the README).")
    st.stop()

data = load_data()

with st.sidebar:
    if os.path.exists("assets/hcss_logo.png"):
        st.image("assets/hcss_logo.png", width=140)
    st.title("Reliability in GENOME")
    st.caption("A journey from the GENOME map to a reliability meter")
    page = st.radio(
        "Navigation",
        ["1 · Starting analysis", "2 · Support clusters", "3 · The map", "4 · Support chains",
         "5 · Case studies", "6 · Escalation ladder", "7 · Limits & next steps"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: GENOME (HCSS Datalab). This app uses free count queries only; "
               "no API credits are spent.")


if page.startswith("1"):
    analysis.render(data)
elif page.startswith("2"):
    clusters_page.render(data)
elif page.startswith("3"):
    map_page.render(data)
elif page.startswith("4"):
    chains_page.render(data)
elif page.startswith("5"):
    case_page.render(data)
else:
    placeholders.render(page)
