"""Page 2: support clusters. Which countries form blocks of mutual support, who bridges them, and where do words and actions differ?

All heavy statistics are precomputed by scripts/build_cluster_results.py; this page reads them. No API calls are made
here and no credits can be spent.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import cluster_tables as ct
import clusters as cl
import geo
from views.common import (FRAMING_NOTE, PALETTE, PRIMARY, Data, centroids_by_id, cluster_stamp, load_cluster_results,
                          load_edges, your_words)

UNCERTAIN, NO_TIES, LEFT_OUT = "#C9D1DE", "#EEF1F6", "#4A4A4A"
STABLE = 0.7


def _dates(period: str, meta: dict) -> tuple:
    if period == "all":
        return meta["since"], meta["latest"]
    p = pd.Period(period, freq="Q")
    return max(str(p.start_time.date()), meta["since"]), min(str(p.end_time.date()), meta["latest"])


def _summary(res: dict, n_countries: int) -> str:
    """The headline in plain words, from the whole-period stored results."""
    cfg, ag = res["configs"], res["agreement"]
    get = lambda layer, cat: cfg[cfg.cid == ct.config_id(layer, cat, "all", False, 0)].iloc[0]   # noqa: E731
    a, w = get("support", "cooperation"), get("words", "cooperation")
    row = ag[(ag.actions == "support") & (ag.period == "all") & (~ag["log"]) & (ag.hubs == 0)].iloc[0]
    cv = res["conflict_vs_clusters"].merge(cfg[["cid", "layer", "period", "log", "hubs"]], on="cid")
    cv = cv[(cv.layer == "support") & (cv.period == "all") & (~cv["log"]) & (cv.hubs == 0)].iloc[0]
    parts = [
        f"Looking at **actions** (reported military, humanitarian and financial aid) between {n_countries} countries since January 2024, "
        f"the countries fall into **{int(a.n_clusters)} blocks** that cooperate far more among themselves than the number of "
        f"reports about each country would lead us to expect. {int(a.n_stable)} of {int(a.n_active)} countries have a firm place: on average they share a block "
        f"with their blockmates in at least 70% of 100 repeat runs. The others are shown as uncertain.",
        f"Looking at **words** (statements, talks, agreements) gives **{int(w.n_clusters)} blocks**, and they only partly match "
        f"the action blocks (agreement {row.ari:.2f} on a scale from 0, unrelated, to 1, identical). Countries that sit with "
        f"others in words but not in actions are candidates for words without action, or for support the news does not report."]
    if cv.within_share >= cv.expected:
        parts.append(f"Adversaries do **not** sit in opposite blocks: {cv.within_share:.0%} of conflict events are between "
                     f"countries of the same cooperation block (chance would give {cv.expected:.0%}). The blocks are largely "
                     f"regional: neighbours both cooperate and fight.")
    else:
        parts.append(f"Adversaries tend to sit in different blocks: only {cv.within_share:.0%} of conflict events are between "
                     f"countries of the same cooperation block (chance would give {cv.expected:.0%}).")
    return " ".join(parts)


def _blocks_figure(labels: dict, colour_idx: np.ndarray, member_cluster: np.ndarray, stability: np.ndarray, dropped: np.ndarray,
                   nodes: list, names: dict, iso: dict, cent: dict, geojson: dict, deviating: set, arcs: pd.DataFrame | None,
                   block_names: dict) -> go.Figure:
    """World map with one colour per block. Uncertain, unclustered and left-out countries have their own greys.
    Black dots mark countries whose words and actions place them in different blocks; arcs show the flows between blocks."""
    rows = []
    for i, c in enumerate(nodes):
        if c not in iso:
            continue
        if dropped[i]:
            label, col = "Left out of this view (biggest country)", LEFT_OUT
        elif member_cluster[i] < 0:
            label, col = "No ties in this period", NO_TIES
        elif not np.nan_to_num(stability[i]) >= STABLE:
            label, col = f"Uncertain (placed differently in more than {int((1 - STABLE) * 100)}% of runs)", UNCERTAIN
        else:
            label, col = block_names[int(member_cluster[i])], PALETTE[int(colour_idx[i]) % len(PALETTE)]
        rows.append({"iso": iso[c], "Country": names[c], "label": label, "col": col,
                     "Block certainty": "" if np.isnan(stability[i]) else f"{stability[i]:.2f}"})
    df = pd.DataFrame(rows)
    cmap = dict(zip(df.label, df.col))
    order = sorted(cmap, key=lambda s: (s.startswith(("Uncertain", "No ties", "Left out")), s))
    fig = px.choropleth_map(df, geojson=geojson, featureidkey="properties.ISO_A3_EH", locations="iso", color="label",
                            color_discrete_map=cmap, category_orders={"label": order}, hover_name="Country",
                            hover_data={"iso": False, "label": True, "Block certainty": True}, map_style="carto-positron",
                            zoom=0.7, center={"lat": 25, "lon": 15}, opacity=0.85)
    if arcs is not None and len(arcs):
        wmax = arcs.events.max()
        for r in arcs.itertuples():
            lats, lons = geo.arc(r.la1, r.lo1, r.la2, r.lo2)
            fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines", showlegend=False,
                                        line=dict(width=1.5 + 6 * (r.events / wmax) ** 0.5, color="rgba(30,30,30,0.55)"),
                                        hoverinfo="text", text=f"{r.src} → {r.dst}: {int(r.events)} events"))
    if deviating:
        pts = [(cent[c][0], cent[c][1], names[c]) for c in deviating if c in cent]
        fig.add_trace(go.Scattermap(lat=[p[0] for p in pts], lon=[p[1] for p in pts], mode="markers", name="Words and actions disagree",
                                    marker=dict(size=11, color="black"), text=[p[2] for p in pts], hoverinfo="text"))
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=10)))
    return fig


def _legend(block_names: dict, members: pd.DataFrame, names: dict) -> None:
    for b, label in block_names.items():
        g = members[members.cluster == b].sort_values("strength", ascending=False)
        st.caption(f"**{label.split(':')[0]}** ({len(g)} countries): " + ", ".join(
            names[r.country_id] + ("*" if not np.nan_to_num(r.stability) >= STABLE else "") for r in g.head(14).itertuples())
            + (" …" if len(g) > 14 else ""))


def render(data: Data) -> None:
    st.title("Step 2 · Support clusters")
    files = load_edges()
    res = load_cluster_results(cluster_stamp())
    if files is None or res is None:
        st.error("The cluster results are not built yet. Run `.venv/bin/python scripts/build_cluster_results.py` "
                 "(about 10 minutes, free, no credits).")
        return
    names, meta = files["names"], res["meta"]
    nodes = list(names)
    ids = {v: k for k, v in names.items()}
    iso = {int(r.country_id): r.iso_alpha3 for r in data.countries.itertuples() if isinstance(r.iso_alpha3, str)}
    cent = centroids_by_id(tuple(iso.items()))
    from views.common import world_geojson
    geojson = world_geojson()
    cfgs, members_all = res["configs"], res["members"]

    your_words("clusters_intro")
    st.success(_summary(res, len(names)))
    st.info(FRAMING_NOTE + " A **block** is a statistical pattern in reported events, not proof of a formal alliance.")
    st.info(
        "**Countries mostly cooperate with their neighbours, not with fixed \"camps\". And what they say doesn't "
        "match what they do.**\n\n"
        "A bit more detail: there are real groups of countries that cooperate more often with each other, but "
        "those groups are regional rather than permanent alliances. A few countries (like the UK and Azerbaijan) "
        "sit in between the groups. And countries that sound alike in their words don't go on to cooperate more "
        "in practice."
    )

    # ── fixed view: actions and words side by side, cooperation, support only, whole period ──
    st.subheader("A · The blocks")
    your_words("clusters_map")
    category = "cooperation"
    layer_of = {"Actions": "support", "Words": "words"}
    quarters = [p for p in pd.period_range(meta["since"], meta["latest"], freq="Q").astype(str)]
    period = "all"
    log = False
    hubs = 0
    resolution = 1.0
    start, end = _dates(period, meta)
    drop_ids = ()
    shown = ["Actions", "Words"]

    def pick(layer):
        cid = ct.config_id(layer, category, period, log, hubs, resolution)
        row = cfgs[cfgs.cid == cid]
        return cid, row.iloc[0], members_all[members_all.cid == cid].set_index("country_id").reindex(nodes).reset_index()

    sel = {lbl: pick(layer_of[lbl]) for lbl in shown}
    colour = {}
    if len(shown) == 2:
        a_mem = sel["Actions"][2].cluster.fillna(-1).astype(int).to_numpy()
        w_mem = sel["Words"][2].cluster.fillna(-1).astype(int).to_numpy()
        colour["Actions"], colour["Words"] = ct.align_pair(a_mem, w_mem)
    else:
        colour[shown[0]] = sel[shown[0]][2].cluster.fillna(-1).astype(int).to_numpy()

    dev = res["deviation"]
    dev = dev[(dev.actions == layer_of["Actions"]) & (dev.period == period) & (dev["log"] == log) & (dev.hubs == hubs)].set_index("country_id") \
        if resolution == 1.0 and category == "cooperation" else pd.DataFrame()
    deviating = set()
    if len(shown) == 2 and len(dev):
        sw, sa = sel["Words"][2].set_index("country_id").stability, sel["Actions"][2].set_index("country_id").stability
        ok = [c for c in dev.index if np.nan_to_num(sw.get(c, np.nan)) >= STABLE and np.nan_to_num(sa.get(c, np.nan)) >= STABLE]
        deviating = set(dev.loc[ok].deviation.sort_values(ascending=False).head(8).index) if ok else set()

    cols = st.columns(len(shown), gap="large")
    for col, lbl in zip(cols, shown):
        cid, row, m = sel[lbl]
        mem = m.cluster.fillna(-1).astype(int).to_numpy()
        stab = m.stability.to_numpy(dtype=float)
        dropped = m.dropped.fillna(False).astype(bool).to_numpy()
        bnames = ct.block_names(m.assign(cluster=mem, strength=m.strength.fillna(0)), names)
        arcs = None
        with col:
            st.markdown(f"##### {lbl}: {category}")
            fig = _blocks_figure({}, colour[lbl], mem, stab, dropped, nodes, names, iso, cent, geojson,
                                 deviating if lbl == shown[-1] and len(shown) == 2 else set(), arcs, bnames)
            st.plotly_chart(fig, width="stretch", key=f"map_{lbl}")
            st.caption(f"{int(row.n_clusters)} blocks; {int(row.n_stable)} of {int(row.n_active)} countries placed with "
                       f"certainty. Beyond chance: {'yes' if row.p <= 0.01 else 'no'} (z = {row.z:.0f}).")
            _legend(bnames, m.assign(cluster=mem), names)
    st.caption("A * after a country means its block is uncertain. " + (
        "Black dots: the countries whose words and actions place them in different blocks." if deviating else ""))

    st.divider()
    st.subheader("How sure are we?")
    your_words("clusters_sure")
    rows = []
    for lbl in ("Actions", "Words"):
        _, r, _ = pick(layer_of[lbl])
        rows.append({"Layer": lbl, "Blocks": int(r.n_clusters), "Countries with a certain block": f"{int(r.n_stable)} of {int(r.n_active)}",
                     "Blocks score (chance)": f"{r.modularity:.2f} ({r.null_mean:.2f})", "Above chance (z)": round(r.z, 1),
                     "p": "<0.001" if r.p <= 0.0011 else round(r.p, 3), "Biggest block (chance)": f"{r.largest_share:.0%} ({r.largest_share_null:.0%})"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "- **Certain** means: on average the country is in the same block as its blockmates in at least 70% of 100 repeat runs, each "
        "on re-drawn events (reports are a sample of what happened) and with a different random start.\n"
        "- **Blocks score vs chance:** compared with 1,000 random networks in which every country keeps exactly the number of "
        "events it gives and receives. A score far above chance means the blocks are real patterns, not a by-product of a few "
        "very busy countries; it does not say they are alliances.")
    st.markdown("**With or without the biggest country**")
    rows = []
    for lbl in ("Actions", "Words"):
        _, _, m0 = pick(layer_of[lbl])
        c1_ = ct.config_id(layer_of[lbl], category, period, log, 1)
        m1 = members_all[members_all.cid == c1_].set_index("country_id").reindex(nodes)
        if len(m1) and period in ("all",) or len(m1):
            a_ = m0.cluster.fillna(-1).astype(int).to_numpy()
            b_ = m1.cluster.fillna(-1).astype(int).to_numpy()
            rows.append({"Layer": lbl, f"Agreement with and without {meta['hubs'][0]}": round(cl.adjusted_rand(a_, b_), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("1 = the same blocks; lower means that the biggest country shaped them.")
    st.markdown("**Changing the resolution (whole period, everyone included)**")
    sen = res["sensitivity"]
    sen = sen[(sen.category == category) & (sen["log"] == log) & sen.layer.isin([layer_of["Actions"], "words"])].copy()
    sen["layer"] = sen.layer.map({"material": "Actions", "support": "Actions", "words": "Words"})
    figs = px.line(sen, x="resolution", y="n_clusters", color="layer", markers=True,
                   labels={"resolution": "Resolution (higher = finer blocks)", "n_clusters": "Number of blocks", "layer": ""})
    figs.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figs, width="stretch", key="sens_chart")
    sen = sen.assign(**{"Agreement with resolution 1": sen.ari_vs_resolution_1.round(2), "Mean certainty": sen.mean_stability.round(2)})
    st.dataframe(sen[["layer", "resolution", "n_clusters", "Mean certainty", "Agreement with resolution 1"]].rename(
        columns={"layer": "Layer", "resolution": "Resolution", "n_clusters": "Blocks"}), hide_index=True, width="stretch")
    st.markdown("**Quarter by quarter**")
    q = cfgs[(cfgs.category == category) & (cfgs["log"] == log) & (cfgs.hubs == min(hubs, 1)) & (cfgs.period != "all") & (cfgs.resolution == 1.0)
             & cfgs.layer.isin([layer_of["Actions"], "words"])].copy()
    q["layer"] = q.layer.map({"material": "Actions", "support": "Actions", "words": "Words"})
    figq = px.bar(q, x="period", y="mean_stability", color="layer", barmode="group", color_discrete_sequence=[PRIMARY, "#E69F00"],
                  labels={"period": "", "mean_stability": "Average certainty of the countries' blocks", "layer": ""})
    figq.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 1]))
    st.plotly_chart(figq, width="stretch", key="quarter_stab")
    st.caption("Quarters hold few events for material cooperation, so its blocks are less certain than for the whole period: "
               "read quarter-by-quarter results as leads, and trust the whole-period blocks more. The last quarter is incomplete.")
