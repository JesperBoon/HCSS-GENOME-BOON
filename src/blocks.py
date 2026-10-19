"""Words vs. actions blocks: do the same groups of countries cooperate in what they say as in
what they do, on the precomputed clusters in data/processed/clusters/?

New module (does not change anything in the existing src/ files, including clusters.py and
cluster_tables.py). Uses the stored results from scripts/build_cluster_results.py - no clustering
is rerun here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import cluster_tables as ct
import metrics

ROOT = Path(__file__).resolve().parent.parent
MEMBERS = ROOT / "data" / "processed" / "clusters" / "members.csv"
LAYERS = {"words": "words", "material": "material"}


def _membership(cid: str, min_stability: float) -> pd.Series:
    """country_id -> cluster (only countries clustered stably enough; others are left out)."""
    m = pd.read_csv(MEMBERS)
    m = m[(m.cid == cid) & (m.cluster >= 0) & (m.stability >= min_stability)]
    return m.set_index("country_id")[["cluster", "stability", "strength"]]


def words_vs_material(category: str = "cooperation", period: str = "all", log: int = 0,
                      hub: int = 0, res: float = 1.0, min_stability: float = 0.7) -> pd.DataFrame:
    """One row per country clustered in EITHER layer: its words block and material block
    (aligned so the same group of countries gets the same label in both), and whether it
    switched blocks between the two."""
    cids = {layer: ct.config_id(layer, category, period, bool(log), hub, res) for layer in LAYERS}
    mem = {layer: _membership(cid, min_stability) for layer, cid in cids.items()}

    ids = sorted(set(mem["words"].index) | set(mem["material"].index))
    idx = {c: i for i, c in enumerate(ids)}
    raw = {layer: -np.ones(len(ids), dtype=int) for layer in LAYERS}
    for layer, m in mem.items():
        for country_id, row in m.iterrows():
            raw[layer][idx[country_id]] = row.cluster

    words_aligned, material_aligned = ct.align_pair(raw["words"], raw["material"])
    countries = metrics.load_countries().set_index("country_id")
    out = pd.DataFrame({"country_id": ids, "cluster_words": words_aligned, "cluster_material": material_aligned})
    out = out[(out.cluster_words >= 0) | (out.cluster_material >= 0)]
    out["name"] = out.country_id.map(countries.name)
    out["iso_alpha3"] = out.country_id.map(countries.iso_alpha3)
    out["switched_block"] = (out.cluster_words >= 0) & (out.cluster_material >= 0) & (out.cluster_words != out.cluster_material)
    return out.reset_index(drop=True)


def blocks_map(world: dict, category: str = "cooperation", period: str = "all", log: int = 0,
               hub: int = 0, res: float = 1.0, min_stability: float = 0.7) -> go.Figure:
    """Dropdown 'words blocks' / 'material blocks'. Same colour = same block; a country that
    switches colour between the two views cooperates with a different group in deeds than in words."""
    d = words_vs_material(category, period, log, hub, res, min_stability)
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]

    fig = go.Figure()
    for layer, col in [("words", "cluster_words"), ("material", "cluster_material")]:
        dd = d[d[col] >= 0]
        colors = [palette[c % len(palette)] for c in dd[col]]
        fig.add_trace(go.Choropleth(
            geojson=world, featureidkey="properties.ISO_A3_EH", locations=dd.iso_alpha3,
            z=dd[col], text=dd.name + " - Block " + (dd[col] + 1).astype(str), visible=(layer == "words"),
            colorscale=[[i / max(len(palette) - 1, 1), c] for i, c in enumerate(palette)],
            showscale=False, hovertemplate="%{text}<extra></extra>"))

    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    showland=True, landcolor="#e6e6e6")
    fig.update_layout(
        title=f"Support blocks by {{layer}}: same colour = same block ({category}, {period})".replace("{layer}", "words"),
        updatemenus=[dict(buttons=[
            dict(label="words blocks", method="update",
                args=[{"visible": [True, False]}, {"title": f"Support blocks by words ({category}, {period})"}]),
            dict(label="material blocks", method="update",
                args=[{"visible": [False, True]}, {"title": f"Support blocks by material actions ({category}, {period})"}]),
        ], x=0, y=1.08, xanchor="left", direction="down")],
        margin=dict(l=0, r=0, t=70, b=0), height=650)
    return fig
