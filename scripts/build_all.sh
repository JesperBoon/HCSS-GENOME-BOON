#!/bin/zsh
# build_all.sh: rebuild every processed data file from the public GENOME API.
#
# FREE: uses only unauthenticated count queries and first pages (no API key, no credits). The edge-count
# step uses a free-only client that refuses any request needing the key. Every step is resumable, so it is
# safe to stop and rerun.
#
# Takes roughly 2.5 hours in total. Run from the project root:   zsh scripts/build_all.sh
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

CONFLICTS=("Russia Ukraine" "Iran Israel" "Israel Palestine" "China USA")

echo "1/9  Monthly counts for 87 countries (about 5 min)"
$PY scripts/build_monthly_counts.py

echo "2/9  Most hostile country pairs (about 4 min)"
$PY scripts/find_conflict_pairs.py

echo "3/9  Quarterly counts per conflict (about 8 min each)"
for pair in $CONFLICTS; do $PY scripts/build_dyad_counts.py ${=pair}; done

echo "4/9  Weekly counts for the 30 biggest players per conflict (about 10 min each)"
for pair in $CONFLICTS; do $PY scripts/build_weekly_counts.py ${=pair}; done

echo "5/9  Every event between the 60 biggest countries, for the support chains (about 8 min)"
$PY scripts/build_edge_counts.py --run

echo "6/9  Text of every event: summary, quote, names, locations, links (about 2 min)"
$PY scripts/build_event_text.py

echo "7/9  Chain statistics and the words-to-aid test (about 15 min)"
$PY scripts/build_chain_results.py

echo "8/9  Support clusters: 310 clusterings, agreement, prediction (about 12 min)"
$PY scripts/build_cluster_results.py

echo "9/9  Which definition of support gives the most stable blocks (about 1 min)"
$PY scripts/compare_variants.py

# Not run here because it spends credits: scripts/verify_slicing.py (about 450 credits, once) compares paid paging with free slicing.

echo "Done. Start the dashboard with:  .venv/bin/streamlit run app.py"
