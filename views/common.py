"""Shared pieces of the dashboard: constants, cached data loaders and small helpers."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import geo
import metrics as mt
from genome_api import GenomeClient
from journey_text import HINTS, TEXT

ROOT = Path(__file__).resolve().parent.parent
PRIMARY = "#003082"      # HCSS blue, also set as primaryColor in .streamlit/config.toml
NO_SIDE = 0.3            # |side score| below this is shown as "no clear side"


@dataclass(frozen=True)
class Data:
    """The data every page needs, loaded once."""
    countries: pd.DataFrame
    monthly: pd.DataFrame
    name_of: dict
    months: list


@dataclass(frozen=True)
class Conflict:
    """The conflict chosen on page 1, plus the weighting switch that applies to all its sections."""
    name: str
    first_name: str
    second_name: str
    first_id: int
    second_id: int
    weighted: bool


def your_words(key: str):
    """Show the author's text for a journey step, if there is one."""
    text = TEXT.get(key, "").strip()
    if text:
        st.markdown(text)


@st.cache_data(show_spinner=False)
def load_all():
    return mt.load_countries(), mt.load_monthly()


@st.cache_data(show_spinner=False)
def world_geojson():
    """Country borders (Natural Earth 1:110m, public domain), keyed on ISO_A3_EH."""
    with open(ROOT / "assets" / "world.geojson") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def recent_events(country_id: int):
    """Newest events where the country is actor or recipient (free page-1 queries)."""
    c = GenomeClient()
    rows = (c.events_page(1, actor_country_id=country_id)["events"]
            + c.events_page(1, recipient_country_id=country_id)["events"])
    seen, out = set(), []
    for e in sorted(rows, key=lambda e: e["event_date"], reverse=True):
        if e["id"] not in seen:
            seen.add(e["id"])
            out.append(e)
    return out



def load_data() -> Data:
    countries, monthly = load_all()
    return Data(countries, monthly, dict(zip(countries.country_id, countries.name)),
                sorted(monthly.month.unique()))


NO_SIDE = 0.3   # |side score| below this is shown as "no clear side"


@st.cache_data(show_spinner=False)
def load_pairs():
    return pd.read_csv(mt.ROOT / "data" / "processed" / "conflict_pairs.csv")


@st.cache_data(show_spinner=False)
def side_scores(conflict: str, first_id: int, second_id: int, quarters: tuple,
                min_n: int = 3, weighted: bool = False):
    return mt.side_table(conflict, first_id, second_id, list(quarters), min_n=min_n, weighted=weighted)


@st.cache_data(show_spinner="Computing rolling scores…")
def rolling_scores(conflict: str, first_id: int, second_id: int, window: int, min_n: int,
                   weighted: bool, stamp: float):
    """`stamp` (file modification time) only exists so the cache refreshes when new weekly data arrives."""
    return mt.rolling_scores(conflict, first_id, second_id, window, min_n, weighted)


def side_chart(df, col, first, second, kind_label, breakdown=False):
    """Horizontal bars: negative = leans to `first`, positive = leans to `second`.

    If `breakdown`, also marks each country's separate words-only and actions-only side
    score on top of its bar (when `df` has `side_verbal`/`side_material` columns) - so a
    combined score that looks moderate can be seen for what it sometimes is: friendly words
    and hostile actions (or the reverse) averaging out to "no clear side"."""
    d = df.dropna(subset=[col]).sort_values(col).copy()
    d["Position"] = pd.cut(d[col], [-3, -NO_SIDE, NO_SIDE, 3],
                           labels=[f"Leans {first}", "No clear side", f"Leans {second}"])
    d["Events toward " + first] = d[f"n_{kind_label}_first"].astype(int)
    d["Events toward " + second] = d[f"n_{kind_label}_second"].astype(int)
    fig = px.bar(
        d, x=col, y="name", orientation="h", color="Position",
        color_discrete_map={f"Leans {first}": "#C62828", "No clear side": "#B8C4D6",
                            f"Leans {second}": PRIMARY},
        hover_data={col: ":.2f", "name": False, "Position": False,
                    "Events toward " + first: True, "Events toward " + second: True},
        labels={col: f"← {first}      side score      {second} →", "name": ""},
    )
    fig.update_layout(height=max(320, 19 * len(d) + 80), margin=dict(l=0, r=10, t=10, b=0),
                      legend=dict(orientation="h", y=1.04, title=""), bargap=0.25)
    fig.add_vline(x=0, line_width=1, line_color="#1A1A1A")

    if breakdown and "side_verbal" in d.columns and "side_material" in d.columns:
        for label, dcol, color in [("Words only", "side_verbal", "#6A1B9A"),
                                   ("Actions only", "side_material", "#EF6C00")]:
            fig.add_trace(go.Scatter(
                x=d[dcol], y=d["name"], mode="markers", name=label,
                marker=dict(symbol="line-ns", size=16, line=dict(width=3, color=color)),
                hovertemplate=f"%{{y}} — {label.lower()}: %{{x:.2f}}<extra></extra>",
            ))
    return fig, len(d)


# ── support-chain data (pages 3 and 4) ────────────────────────────────────────

EDGES_DIR = ROOT / "data" / "processed" / "edges"
CHAINS_DIR = ROOT / "data" / "processed" / "chains"

FRAMING_NOTE = (
    "**How to read this.** GENOME reports *events* (\"A gives aid to B\" on a date). It has no amounts: no money, no "
    "weapons. What you see here are **reported paths** of events, in the news, ordered only by time window. They are "
    "not proven flows, and closeness in time is not causation. *Hidden* means *not observed in this English-language "
    "newswire corpus*, not secret. Organisations such as the EU, NATO and the UN have no country id in the data, so "
    "they cannot appear. Meetings where two countries are both coded as actors have no recipient and are not shown."
)


@st.cache_data(show_spinner="Loading event data…")
def load_edges():
    """Events, their country pairs and the country names from scripts/build_edge_counts.py (or None)."""
    if not (EDGES_DIR / "events.csv").exists():
        return None
    meta = json.loads((EDGES_DIR / "meta.json").read_text())
    return {"events": pd.read_csv(EDGES_DIR / "events.csv"), "edges": pd.read_csv(EDGES_DIR / "event_edges.csv"),
            "leaves": pd.read_csv(EDGES_DIR / "leaf_counts.csv"), "meta": meta,
            "names": {int(k): v for k, v in meta["countries"].items()}}


def chain_stamp() -> float:
    """Modification time of the precomputed chain results; part of the cache key so new results are picked up."""
    f = CHAINS_DIR / "chains.csv"
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_data(show_spinner="Loading chain results…")
def load_chain_results(stamp: float):
    """Precomputed tables from scripts/build_chain_results.py (or None while they do not exist)."""
    if not stamp:
        return None
    out = {name: pd.read_csv(CHAINS_DIR / f"{name}.csv") for name in
           ("chains", "summary", "brokers", "cycles", "prediction", "classes")}
    out["meta"] = json.loads((CHAINS_DIR / "meta.json").read_text())
    return out


@st.cache_data(show_spinner=False)
def centroids_by_id(country_ids_and_iso: tuple):
    """{country id: (lat, lon)} from the world borders; countries missing from the map file are left out."""
    by_iso = geo.country_centroids(world_geojson())
    return {cid: by_iso[iso] for cid, iso in country_ids_and_iso if iso in by_iso}


@st.cache_data(show_spinner="Fetching the reported events…", ttl=6 * 3600)
def live_events(actor_id: int, recipient_id: int, event_type: str, since: str, until: str):
    """The events behind a link, from the FREE first page of the API (newest first, at most 250).
    Returns (events, total). The client is free-only: anything that would cost credits raises."""
    c = GenomeClient(free_only=True)
    page = c.events_page(1, actor_country_id=actor_id, recipient_country_id=recipient_id, event_type=event_type,
                         event_date_from=since, event_date_to=until)
    return page["events"], page["total"]


# ── support-cluster data (page 2) ─────────────────────────────────────────────

CLUSTERS_DIR = ROOT / "data" / "processed" / "clusters"


def cluster_stamp() -> float:
    f = CLUSTERS_DIR / "configs.csv"
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_data(show_spinner="Loading cluster results…")
def load_cluster_results(stamp: float):
    """Tables from scripts/build_cluster_results.py (or None while they do not exist)."""
    if not stamp:
        return None
    out = {n: pd.read_csv(CLUSTERS_DIR / f"{n}.csv") for n in
           ("configs", "members", "flows", "conflict_vs_clusters", "sensitivity", "agreement", "deviation", "prediction")}
    out["meta"] = json.loads((CLUSTERS_DIR / "meta.json").read_text())
    npz = np.load(CLUSTERS_DIR / "coassign.npz")
    out["coassign"] = {k: npz[k] for k in npz.files}
    return out


@st.cache_data(show_spinner="Reading the AID events…")
def marked_events():
    """Stored events with every AID event classified; AID that is military, humanitarian or financial has the type AID_SUPPORT."""
    import support_events as se
    return se.load_events()


@st.cache_data(show_spinner=False)
def pair_counts(layer: str, category: str, since: str, until: str, drop_ids: tuple = ()):
    """(n, n) event counts between the stored countries for one layer ('material', 'support' or 'words'), category and period
    (countries in drop_ids zeroed). 'support' counts only AID that is military, humanitarian or financial."""
    import clusters as cl
    import chains as ch
    import support_events as se
    files = load_edges()
    nodes = list(files["names"])
    W = cl.weights_matrix(ch.pair_weights(marked_events(), files["edges"], se.types_for(layer, category), since, until), nodes)
    return cl.without(W, [nodes.index(c) for c in drop_ids]) if drop_ids else W


PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#7A6FBA", "#8C564B"]   # colour-blind safe


def style_strip(text: pd.DataFrame):
    """Colour the cells of a membership strip by group number; '?' and '–' stay grey."""
    def one(v):
        if str(v).isdigit():
            return f"background-color: {PALETTE[(int(v) - 1) % len(PALETTE)]}; color: white"
        return "background-color: #EEF1F6; color: #5A6B85"
    return text.style.map(one)
