# Project: Road to Reliability

## Deliverable
A narrative Jupyter notebook (`Road_to_Reliability.ipynb`, plus an HTML export) about
the gap between what countries say and what they do in GENOME event data
(verbal vs. material events). This is for the HCSS Datalab technical assignment.
The notebook is the main deliverable. The dashboard (tagged `dashboard-v1`) is
secondary and only serves as an optional "try it yourself" companion.

## Language
Everything is in English: notebook text, code comments, docstrings, README.

## Rules of collaboration
- The user writes all narrative text (markdown cells). Never write or rewrite it
  unless explicitly asked.
- You write code cells only on request. Keep each cell short (under ~30 lines);
  put heavy logic in `src/` and import it.
- Reuse existing modules: `src/genome_api.py`, `src/edge_counts.py`,
  `src/taxonomy.py`, `src/metrics.py`, `src/geo.py`, `assets/world.geojson`,
  and the data in `data/processed/`.
- Only use `src/clusters.py`, `src/cluster_tables.py` and `src/hidden.py` when
  the user asks for them.
- Do not modify existing files in `src/` without saying so first.
- Do not build extra features, pages or tests. If scope is unclear, ask.
- Never spend API credits without asking first.
