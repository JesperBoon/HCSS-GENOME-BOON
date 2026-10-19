"""Page 2: support clusters. Which countries form blocks of mutual support, who bridges them, and where do words and actions differ?

All heavy statistics are precomputed by scripts/build_cluster_results.py; this page reads them. No API calls are made
here and no credits can be spent.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import chains as ch
import cluster_tables as ct
import clusters as cl
import geo
import hidden as hd
from views.common import (FRAMING_NOTE, PALETTE, PRIMARY, Data, centroids_by_id, cluster_stamp, load_cluster_results,
                          load_edges, marked_events, pair_counts, style_strip, your_words)

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


def _block_arcs(flows: pd.DataFrame, member_cluster: np.ndarray, stability: np.ndarray, strength: np.ndarray, nodes: list,
                cent: dict, block_names: dict, top: int = 8) -> pd.DataFrame:
    """Arcs between the centres of blocks for the biggest flows. The centre of a block is the busiest-weighted mean position of
    its stable members."""
    pos = {}
    for b in block_names:
        idx = [i for i in np.flatnonzero(member_cluster == b) if nodes[i] in cent and np.nan_to_num(stability[i]) >= STABLE]
        if idx:
            w = strength[idx] + 1e-9
            pos[b] = (float(np.average([cent[nodes[i]][0] for i in idx], weights=w)),
                      float(np.average([cent[nodes[i]][1] for i in idx], weights=w)))
    f = flows[(flows.from_cluster != flows.to_cluster) & flows.from_cluster.isin(pos) & flows.to_cluster.isin(pos)]
    f = f[f.events > 0].sort_values("events", ascending=False).head(top)
    return pd.DataFrame({"src": f.from_cluster.map(lambda b: block_names[b].split(":")[0]),
                         "dst": f.to_cluster.map(lambda b: block_names[b].split(":")[0]), "events": f.events,
                         "la1": [pos[b][0] for b in f.from_cluster], "lo1": [pos[b][1] for b in f.from_cluster],
                         "la2": [pos[b][0] for b in f.to_cluster], "lo2": [pos[b][1] for b in f.to_cluster]})


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

    # ── controls ──────────────────────────────────────────────────────────────
    st.subheader("A · The blocks")
    your_words("clusters_map")
    c1, c2, c3, c4 = st.columns([2, 3, 2, 2], gap="large")
    view = c1.radio("Show", ["Actions and words side by side", "Actions only", "Words only"])
    category = c1.radio("Kind", ["Cooperation", "Conflict"], horizontal=True).lower()
    support_only = c1.radio("Actions counted as", ["Support only", "All material cooperation"], disabled=category == "conflict",
                            help="Support only keeps aid that is military, humanitarian or financial and leaves out exchanges of prisoners and "
                                 "bodies, evacuations, trade measures and ceremonies that GENOME also codes as AID. That is the more robust "
                                 "choice. 'All material cooperation' is the paper's definition (AID, cooperation, de-escalation).") == "Support only"
    layer_of = {"Actions": "support" if support_only and category == "cooperation" else "material", "Words": "words"}
    quarters = [p for p in pd.period_range(meta["since"], meta["latest"], freq="Q").astype(str)]
    options = ["Whole period"] + [q + (" (incomplete)" if q in meta["partial_periods"] else "") for q in quarters]
    period_label = c2.select_slider("Period", options=options, value="Whole period",
                                    help="Quarters have fewer events, so their blocks are less certain: look at the grey countries.")
    period = "all" if period_label == "Whole period" else period_label.split(" ")[0]
    log = c3.radio("Counting", ["Event counts", "Damped (log)"], help="Damped counting stops a few very busy pairs from "
                   "deciding the blocks.") == "Damped (log)"
    hub_labels = {0: "Everyone", 1: f"Without {meta['hubs'][0]}", 3: "Without " + ", ".join(meta["hubs"][:3])}
    hubs_pick = c4.radio("Biggest countries", [hub_labels[0], hub_labels[1], hub_labels[3]],
                         help="The most-mentioned countries tie everything together. Leaving them out shows whether the "
                              "blocks depend on them. Leaving out three is available for the whole period only.")
    hubs = {v: k for k, v in hub_labels.items()}[hubs_pick]
    if hubs == 3 and period != "all":
        st.caption("Leaving out three countries is only computed for the whole period; showing the version without "
                   f"{meta['hubs'][0]} instead.")
        hubs = 1
    resolution = 1.0
    if period == "all" and hubs == 0:
        resolution = c2.select_slider("Resolution", options=[0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
                                      help="Higher = more, smaller blocks. 'How sure are we?' shows how the result changes.")
    start, end = _dates(period, meta)
    drop_ids = tuple(ids[n] for n in meta["hubs"][:hubs])

    def pick(layer):
        cid = ct.config_id(layer, category, period, log, hubs, resolution)
        row = cfgs[cfgs.cid == cid]
        return cid, row.iloc[0], members_all[members_all.cid == cid].set_index("country_id").reindex(nodes).reset_index()

    shown = {"Actions and words side by side": ["Actions", "Words"], "Actions only": ["Actions"], "Words only": ["Words"]}[view]
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
    show_flows = st.toggle("Show flows between blocks", value=False)

    cols = st.columns(len(shown), gap="large")
    for col, lbl in zip(cols, shown):
        cid, row, m = sel[lbl]
        mem = m.cluster.fillna(-1).astype(int).to_numpy()
        stab = m.stability.to_numpy(dtype=float)
        dropped = m.dropped.fillna(False).astype(bool).to_numpy()
        bnames = ct.block_names(m.assign(cluster=mem, strength=m.strength.fillna(0)), names)
        arcs = None
        if show_flows:
            fl = res["flows"]
            fl = fl[(fl.cid == cid) & (fl.kind == "own")]
            arcs = _block_arcs(fl, mem, stab, m.strength.fillna(0).to_numpy(), nodes, cent, bnames)
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

    tabs = st.tabs(["Brokers and flows", "Words versus actions", "How sure are we?", "Over time", "Chains inside and between blocks"])

    # ── brokers and flows ─────────────────────────────────────────────────────
    with tabs[0]:
        lbl = st.radio("Layer", shown, horizontal=True, key="bf_layer") if len(shown) > 1 else shown[0]
        cid, row, m = sel[lbl]
        mm = m.assign(cluster=m.cluster.fillna(-1).astype(int), strength=m.strength.fillna(0))
        bnames = ct.block_names(mm, names)
        W = pair_counts(layer_of[lbl], category, start, end, drop_ids)
        st.markdown("##### Brokers: countries that tie blocks together")
        st.markdown("The **bridge score** is 0 when all of a country's ties are inside its own block and approaches 1 when they "
                    "are spread evenly over several blocks. It is only meaningful if it is higher than a random network with "
                    "the same number of events per country would give, so a country is marked as standing out only then.")
        your_words("clusters_brokers")
        bt = ct.broker_table(W, mm.dropna(subset=["participation"]), nodes, names, STABLE)
        if bt.empty:
            st.info("No country with a certain block and a bridge score here.")
        else:
            show = bt.drop(columns=["country_id", "p"]).copy()
            show["Ties outside own block"] = 100 * show["Ties outside own block"]
            show["Block"] = show.Block.map(lambda s: bnames.get(int(s.split(" ")[1]) - 1, s).split(":")[0])
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "Participation": st.column_config.NumberColumn("Bridge score", format="%.2f"),
                "Z": st.column_config.NumberColumn("Above chance (z)", format="%.1f",
                                                   help="How many standard deviations above a random network with the same numbers of events."),
                "Ties outside own block": st.column_config.ProgressColumn("Ties outside own block", min_value=0, max_value=100, format="%.0f%%")})
            st.caption("Only countries whose block is certain are listed. 'Ties mostly to' names the two other blocks it is tied to most.")
        st.markdown("##### Flows between blocks")
        fl = res["flows"]
        own = fl[(fl.cid == cid) & (fl.kind == "own")]
        if own.empty:
            st.caption("No flows.")
        else:
            piv = own.pivot_table(index="from_cluster", columns="to_cluster", values="events", fill_value=0)
            lab = {b: bnames[b].split(":")[0] for b in piv.index if b in bnames}
            piv = piv.rename(index=lab, columns=lab)
            figh = px.imshow(piv, text_auto=True, color_continuous_scale="Blues", aspect="auto", labels=dict(x="To", y="From", color="Events"))
            figh.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
            st.plotly_chart(figh, width="stretch", key="flows_own")
            st.caption("Events from the block on the left to the block at the bottom. The diagonal is inside a block.")
        if category == "cooperation":
            st.markdown("##### Do adversaries sit in different blocks?")
            cv = res["conflict_vs_clusters"]
            cv = cv[cv.cid == cid]
            if len(cv):
                r = cv.iloc[0]
                direction = "less" if r.within_share < r.expected else "more"
                st.markdown(f"Of **{int(r.events):,}** conflict events between these countries, **{r.within_share:.0%}** are between countries "
                            f"of the **same** cooperation block. If blocks were handed out at random it would be {r.expected:.0%}, so conflict "
                            f"is {direction} often inside blocks than chance"
                            + (" (the difference is large: the blocks are regional, neighbours cooperate and fight)." if direction == "more" and r.z > 2 else "."))
            cf = fl[(fl.cid == cid) & (fl.kind == "conflict_over_cooperation_clusters")]
            if len(cf):
                piv = cf.pivot_table(index="from_cluster", columns="to_cluster", values="events", fill_value=0)
                lab = {b: bnames[b].split(":")[0] for b in piv.index if b in bnames}
                figc = px.imshow(piv.rename(index=lab, columns=lab), text_auto=True, color_continuous_scale="Oranges", aspect="auto",
                                 labels=dict(x="Against", y="From", color="Conflict events"))
                figc.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
                st.plotly_chart(figc, width="stretch", key="flows_conflict")
                st.caption("Conflict events, by the cooperation blocks of the two countries involved.")

    # ── words versus actions ──────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("##### Do words and actions form the same blocks?")
        your_words("clusters_words")
        ag = res["agreement"]
        ag = ag[(ag.actions == layer_of["Actions"]) & (ag.period == period) & (ag["log"] == log) & (ag.hubs == hubs)]
        if ag.empty or category != "cooperation":
            st.info("Comparison available for cooperation, for the chosen period and counting." if category != "cooperation" else "Not computed for this setting.")
        else:
            r = ag.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Agreement (0 = unrelated, 1 = identical)", f"{r.ari:.2f}",
                      help="Adjusted Rand index between the words blocks and the action blocks, corrected for chance.")
            k2.metric("What chance gives", f"{r.ari_null_mean:.2f}", help="Average agreement after shuffling which country is in which action block.")
            k3.metric("Countries compared", f"{int(r.n_countries)}")
            lo = (f" When the events are re-drawn and each layer is clustered once, the agreement ranges from {r.ari_boot_low:.2f} to "
                  f"{r.ari_boot_high:.2f} (95%); single clusterings are noisier than the consensus used for the headline number, so that "
                  f"range runs lower." if pd.notna(r.get("ari_boot_low")) else "")
            st.markdown(f"The agreement is clearly above chance (p = {r.ari_p:.3f}) but far from complete.{lo} Using only the "
                        f"{int(r.n_stable_both)} countries whose block is certain in both layers it is {r.ari_stable_only:.2f}. "
                        "Both layers are clustered the same way; the difference is in which countries cooperate in words and in actions.")
            wm, am = sel.get("Words"), sel.get("Actions")
            if wm is None or am is None:
                wm, am = pick("words"), pick(layer_of["Actions"])
            mw, ma = wm[2], am[2]
            memw, mema = mw.cluster.fillna(-1).astype(int).to_numpy(), ma.cluster.fillna(-1).astype(int).to_numpy()
            stw, sta = mw.stability.to_numpy(float), ma.stability.to_numpy(float)
            classes = hd.pair_classes(memw, mema, stw, sta, STABLE)
            counts = classes["class"].value_counts()
            q1, q2, q3 = st.columns(3)
            q1.metric("Same block in words and in actions", f"{counts.get(hd.CLASS_SAME_BOTH, 0):,}", help="Pairs of countries, both certain.")
            q2.metric("Same block in words only", f"{counts.get(hd.CLASS_SAME_WORDS, 0):,}", help="Candidate latent ties, or empty words.")
            q3.metric("Same block in actions only", f"{counts.get(hd.CLASS_SAME_ACTIONS, 0):,}", help="Silent support: shared actions without shared words.")
            st.caption(f"{counts.get(hd.CLASS_UNCERTAIN, 0):,} pairs involve a country whose block is not certain in one of the layers and are left out.")
            st.markdown("##### Countries whose words and actions disagree most")
            d = dev.reset_index()[["country_id", "words_not_actions", "actions_not_words", "deviation"]] if len(dev) else pd.DataFrame()
            if len(d):
                d = d[d.country_id.map(lambda c: np.nan_to_num(mw.set_index("country_id").stability.get(c, np.nan)) >= STABLE
                                       and np.nan_to_num(ma.set_index("country_id").stability.get(c, np.nan)) >= STABLE)]
                d = d.assign(Country=d.country_id.map(names)).sort_values("deviation", ascending=False).head(12)
                d[["words_not_actions", "actions_not_words"]] = 100 * d[["words_not_actions", "actions_not_words"]]
                st.dataframe(d[["Country", "words_not_actions", "actions_not_words"]].rename(columns={
                    "words_not_actions": "Words partners that are not action partners",
                    "actions_not_words": "Action partners that are not words partners"}), hide_index=True, width="stretch",
                    column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.0f%%") for c in
                                   ("Words partners that are not action partners", "Action partners that are not words partners")})
                st.caption("Only countries with a certain block in both layers. 100% = none of its partners in one layer is a partner in the other.")
            st.markdown("##### Candidate latent ties")
            st.markdown("Pairs that are in the same **words** block, are in different **action** blocks, and have cooperative words "
                        "directly between them. Either the words are empty, or there is support the English-language news does not "
                        "report. We cannot tell which; this list is where to look.")
            Ww = pair_counts("words", "cooperation", start, end, drop_ids)
            lat = ct.candidate_latent_pairs(Ww, memw, mema, stw, sta, nodes, names, STABLE)
            if lat.empty:
                st.info("No such pair among the countries with a certain block.")
            else:
                st.dataframe(lat, hide_index=True, width="stretch")
        st.markdown("##### Do words blocks predict later material cooperation?")
        pr = res["prediction"]
        pr = pr[(pr["log"] == log) & (pr.hubs == min(hubs, 1)) & pr.ok.astype(bool)]
        p1, p2 = st.columns(2)
        mat = p1.radio("Material cooperation is", ["support", "aid", "material"], format_func=lambda v: {
            "support": "Support only (military, humanitarian, financial aid)", "aid": "All AID",
            "material": "Aid, cooperation, de-escalation"}[v], key="pr_mat")
        new_only = p2.radio("Pairs", [False, True], format_func=lambda b: "Only pairs without earlier material cooperation (a NEW tie)" if b else "All pairs", key="pr_new")
        row = pr[(pr.material == mat) & (pr.only_new == new_only) & (pr.outcome_min == 1)]
        st.markdown(f"Words blocks are formed in the first two thirds of the period (until {meta['split_date']}); the outcome is a material "
                    "tie between the two countries in the last third. Each pair in a shared block is compared with the most similar "
                    "pair that does not share one (same activity of both countries and of the pair).")
        if row.empty:
            st.info("Not computed for this setting.")
        else:
            r = row.iloc[0]
            fig = go.Figure(go.Bar(x=["Pairs in the same words block", "Most similar pairs not in one block", "All other pairs"],
                                   y=[100 * r.rate_treated, 100 * r.rate_matched, 100 * r.rate_all_untreated],
                                   marker_color=[PRIMARY, "#E69F00", "#C9D1DE"],
                                   text=[f"{100 * v:.0f}%" for v in (r.rate_treated, r.rate_matched, r.rate_all_untreated)], textposition="outside"))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                              yaxis=dict(title="% with material cooperation later", range=[0, max(100 * r.rate_treated, 5) * 1.3]))
            st.plotly_chart(fig, width="stretch", key="pred_bar")
            sig = r.ci_low > 0
            st.markdown(f"**{int(r.n_treated)}** pairs. Gap after matching: **{100 * r.difference:+.1f} points** (95% interval "
                        f"{100 * r.ci_low:+.1f} to {100 * r.ci_high:+.1f}). "
                        + ("The interval stays above zero, so words blocks do go with later material ties; the effect is small."
                           if sig else "The interval includes zero: no firm evidence that words blocks predict material ties here. That is a result too."))
            st.caption(f"Balance after matching: at most {r.max_abs_smd_after:.2f} standardised units (below 0.1 is very good).")

    # ── how sure are we ───────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("##### How sure are we?")
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

    # ── over time ─────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("##### Do countries change block over time?")
        your_words("clusters_time")
        t1, t2, t3 = st.columns(3)
        lay = t1.radio("Layer", ["Actions", "Words"], index=1, horizontal=True, key="tl_layer",
                       help="Words are far denser than actions, so their quarterly blocks are more certain.")
        only_stable = t2.toggle("Only countries with a certain block", value=True, key="tl_stable")
        thr = t3.slider("Certainty needed", 0.3, 0.9, STABLE, 0.05, key="tl_thr",
                        help="Lower it to follow more countries, at the price of less certain blocks.")
        if lay == "Actions":
            st.caption("Material cooperation is sparse in a single quarter, so for actions most countries are uncertain in most quarters. "
                       "Lower the certainty, or use the whole-period blocks above, for a firmer picture.")
        th = min(hubs, 1)
        mbp, stab_cols = {}, {}
        for p in quarters:
            m = members_all[members_all.cid == ct.config_id(layer_of[lay], category, p, log, th)]
            if len(m):
                mbp[p] = m
                stab_cols[p] = m.set_index("country_id").stability.reindex(nodes)
        aligned = ct.membership_over_time(mbp, nodes)
        stab_df = pd.DataFrame(stab_cols)
        labels, links = ct.timeline_sankey(aligned, stab_df if only_stable else None, thr if only_stable else 0.0)
        if not links:
            st.info("Not enough certain countries to follow over time.")
        else:
            node_lab = [f"{p} · Group {b + 1}" for p, b in labels]
            node_col = [PALETTE[b % len(PALETTE)] for _, b in labels]
            x = [quarters.index(p) / max(len(quarters) - 1, 1) * 0.98 + 0.01 for p, _ in labels]
            figt = go.Figure(go.Sankey(arrangement="snap", node=dict(label=node_lab, color=node_col, pad=10, thickness=12, x=x, y=[0.5] * len(labels)),
                                       link=dict(source=[s for s, _, _ in links], target=[t for _, t, _ in links], value=[n for _, _, n in links],
                                                 color=[node_col[s].replace("#", "#") + "55" for s, _, _ in links])))
            figt.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figt, width="stretch", key="timeline_sankey")
            st.caption("Each column is a quarter, each band the countries that stay in a group or move to another. Group numbers are matched from one "
                       "quarter to the next by who is in them; the colours follow the same groups. The last quarter is incomplete.")
        changes = ct.block_changes(aligned, stab_df, thr)
        top = changes[changes.comparisons >= 4].sort_values(["share", "changes"], ascending=False).head(10)
        st.markdown("**Countries that change block most often** (of the quarter-to-quarter comparisons in which their block is certain)")
        if top.empty:
            st.info("No country has a certain block in enough quarters to say. For actions most quarters hold too few events; "
                    "lower the certainty above, or look at words.")
        else:
            st.dataframe(pd.DataFrame({"Country": top.index.map(names), "Changes": top.changes, "Comparisons": top.comparisons,
                                       "Share": 100 * top.share}), hide_index=True, width="stretch",
                         column_config={"Share": st.column_config.ProgressColumn("Share of quarters", min_value=0, max_value=100, format="%.0f%%")})
        picks = st.multiselect("Follow countries", sorted(names.values()), default=[n for n in ("USA", "Russia", "Ukraine", "Israel", "Iran", "China") if n in names.values()])
        if picks:
            text = ct.strip_labels(aligned, stab_df, [ids[n] for n in picks], names, thr)
            st.dataframe(style_strip(text), width="stretch")
            st.caption("Number = group, ? = uncertain block, – = no ties in that quarter.")

    # ── chains inside and between blocks ──────────────────────────────────────
    with tabs[4]:
        st.markdown("##### Chains inside and between blocks")
        st.markdown("A chain is A supports B, and B supports C, in the same time window or the next one. We only know the order at the "
                    "level of windows, not days. For the statistics of single chains see page 4, 'Support chains'.")
        your_words("clusters_chains")
        if category != "cooperation" or "Actions" not in shown:
            st.info("Chains are shown for action cooperation. Choose **Cooperation** and a view that includes **Actions**.")
        else:
            cid, row, m = sel["Actions"]
            mem = m.cluster.fillna(-1).astype(int).to_numpy()
            bn = ct.block_names(m.assign(cluster=mem, strength=m.strength.fillna(0)), names)
            c1_, c2_, c3_ = st.columns(3)
            x_b = c1_.selectbox("From block", list(bn), format_func=lambda b: bn[b])
            y_opt = c2_.selectbox("To block", [None] + list(bn), format_func=lambda b: "Inside the same block" if b is None else bn[b])
            kinds = c3_.radio("Support counted as", ["Support only", "All AID", "Aid, cooperation, de-escalation"],
                              index=0 if layer_of["Actions"] == "support" else 2)
            if kinds == "Support only":
                ev_c, types = marked_events(), ["AID_SUPPORT"]
            elif kinds == "All AID":
                ev_c, types = files["events"], ["AID"]
            else:
                ev_c, types = files["events"], None
            g = ch.build_graph(ev_c, files["edges"], files["leaves"], nodes, pd.Timestamp(meta["since"]).date(),
                               pd.Timestamp(end).date(), 4, "material", "cooperation", event_types=types)
            lo = max(0, (pd.Timestamp(start) - pd.Timestamp(meta["since"])).days // 28)
            hi = (pd.Timestamp(end) - pd.Timestamp(meta["since"])).days // 28 + 1
            tbl = cl.chains_between_clusters(g.counts[..., lo:hi], nodes, mem, x_b, y_opt, names)
            if tbl.empty:
                st.info("No chain between these blocks in this period.")
            else:
                st.dataframe(tbl[["path", "weight", "same_window", "next_window"]].rename(columns={
                    "path": "Chain", "weight": "Weight", "same_window": "Same window", "next_window": "Next window"}),
                    hide_index=True, width="stretch")
                st.caption("Weight is the weaker of the two links, summed over 4-week windows. 'Next window': the second link comes after the first. "
                           "These are reported paths, not proven flows, and many are what busy countries produce by chance.")
