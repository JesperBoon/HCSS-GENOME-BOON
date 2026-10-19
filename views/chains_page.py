"""Page 4: support chains. Who passes material support on to whom, and is that more than volume alone?

Sections: A the map of reported support, B are the chains more than volume, C do words come before support.
Everything is built from free count and page-1 data; the heavy statistics are precomputed by
scripts/build_chain_results.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import chains as ch
import geo
import hidden as hd
import taxonomy as tx
from views.common import (FRAMING_NOTE, PRIMARY, Data, centroids_by_id, chain_stamp, load_chain_results,
                          load_edges, your_words)

ORANGE = "#E67E22"
GREY = "#C5CDD9"

# What the user can pick on the map: (layer, category) -> {label: event types}
SELECTIONS = {
    ("material", "cooperation"): {"Aid only": ["AID"], "All material cooperation (aid, joint projects, de-escalation)": tx.MATERIAL_COOP},
    ("material", "conflict"): {"Assaults only": ["ASSAULT"], "All material conflict": tx.MATERIAL_CONF},
    ("words", "cooperation"): {"All cooperative words (talks, agreements, support, concessions)": tx.VERBAL_COOP},
    ("words", "conflict"): {"All hostile words (requests, accusations, rejections, threats)": tx.VERBAL_CONF},
}


def _plain_summary(res: dict, n_countries: int, n_events: int, end: str) -> str:
    """A few sentences without jargon, with the numbers read from the precomputed tables."""
    s = res["summary"]
    base = s[(s.definition == "aid") & (s.kind == "support") & (s.window_weeks == 4)]
    parts = [f"Between January 2024 and {end}, the news reports **{n_events:,} aid events** between {n_countries} countries."]
    if len(base) == 2:
        sh, ma = base[base["null"] == "shuffle"].iloc[0], base[base["null"] == "matched"].iloc[0]
        n_sh, n_ma = int(sh["q_le_0.10"]), int(ma["q_le_0.10"])
        if n_sh == 0 and n_ma == 0:
            verdict = ("**None** of them stands out once we allow for how much each country is in the news, "
                       "and none does when we also allow for weeks in which everything is busy at once.")
        else:
            verdict = (f"After allowing for how much each country is in the news, **{n_sh}** of these chains stand "
                       f"out; after also allowing for weeks in which everything is busy at once, **{n_ma}** remain.")
        parts.append(f"In **{int(ma.tested):,}** cases a country that is reported receiving aid is also reported "
                     f"giving aid to a third country within weeks. Most of that is what busy countries produce by "
                     f"chance. {verdict}")
    p = res["prediction"]
    row = p[(p.material == "aid") & p.only_new & (p.k_verbal == 3) & (p.outcome_min == 1) & p.ok.astype(bool)]
    if len(row):
        r = row.iloc[0]
        parts.append(
            f"Cooperative words are followed by aid more often than chance suggests: country pairs with at least 3 cooperative statements or talks in the "
            f"first two thirds of the period received a **new** aid link in the last third in **{r.rate_treated:.0%}** of "
            f"cases, against **{r.rate_matched:.0%}** for comparable pairs (similar activity) without such words: "
            f"{100 * r.difference:+.0f} points, 95% interval {100 * r.ci_low:+.0f} to {100 * r.ci_high:+.0f}.")
    return " ".join(parts)


def _arcs_figure(pairs: pd.DataFrame, cent: dict, names: dict, color: str, up: pd.DataFrame | None,
                 down: pd.DataFrame | None, top: int) -> tuple:
    """Directed arcs on a world map. Direction is read from the curve (it bows to the left of travel) and from
    the large dot at the receiving end. Returns (figure, number of arcs drawn, arcs skipped for lack of a position)."""
    pairs = pairs.sort_values("weight", ascending=False).head(top)
    pairs = pairs[pairs.actor.isin(cent) & pairs.recipient.isin(cent)]
    fig = go.Figure()
    wmax = max(pairs.weight.max(), 1) if len(pairs) else 1
    up_keys = set(zip(up.actor, up.recipient)) if up is not None and len(up) else set()
    down_keys = set(zip(down.actor, down.recipient)) if down is not None and len(down) else set()
    focus_on = bool(up_keys or down_keys)
    for r in pairs.itertuples():
        (la1, lo1), (la2, lo2) = cent[r.actor], cent[r.recipient]
        lats, lons = geo.arc(la1, lo1, la2, lo2)
        key = (r.actor, r.recipient)
        col = "#0B7A75" if key in down_keys else (PRIMARY if key in up_keys else color)
        faded = focus_on and key not in up_keys and key not in down_keys
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines", showlegend=False,
            line=dict(width=0.6 + 5.4 * (r.weight / wmax) ** 0.5, color=GREY if faded else col),
            opacity=0.5 if faded else 0.85, hoverinfo="text",
            text=f"{names[r.actor]} → {names[r.recipient]}: {int(r.weight)} reported events"))
    inflow = pairs.groupby("recipient").weight.sum()
    fig.add_trace(go.Scattergeo(
        lat=[cent[c][0] for c in inflow.index], lon=[cent[c][1] for c in inflow.index], mode="markers",
        marker=dict(size=4 + 14 * (inflow / max(inflow.max(), 1)) ** 0.5, color=color, opacity=0.9,
                    line=dict(color="white", width=1)),
        text=[f"{names[c]}: {int(w)} events received" for c, w in inflow.items()], hoverinfo="text", showlegend=False))
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#DDE3EC", landcolor="#F3F5F9",
                    showocean=True, oceancolor="#FFFFFF", showframe=False, coastlinecolor="#DDE3EC")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    return fig, len(pairs)


def _tests_section(res: dict, names: dict, nodes: list) -> None:
    """Section B: are the chains more than volume alone? Precomputed tables, selectable settings."""
    st.subheader("B · Are these chains more than busy countries doing busy things?")
    st.markdown(
        "A big country is in the news all the time, so chains through it appear **by volume alone**. For every chain "
        "we compare its weight with what the same events would give if their timing were moved around at random, and "
        "we only call a chain interesting when it stands out from that. Two versions of the comparison:\n\n"
        "- **Volume only:** each country pair's events are moved to any week.\n"
        "- **Volume and shared busy weeks:** events can only move to weeks that are about equally busy for everyone, "
        "so a war week in which everything happens at once does not count as coordination.\n\n"
        "With over a thousand chains tested, some always look special by luck, so the *remaining* counts below are "
        "corrected for that (false discovery rate 10%).")
    your_words("chains_stats")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.radio("What follows what", ["support", "to_conflict"], horizontal=False,
                    format_func=lambda k: {"support": "Aid, then more aid", "to_conflict": "Aid, then an attack"}[k])
    definition = c2.radio("Aid counted as", ["aid", "material"], horizontal=False,
                          format_func=lambda d: {"aid": "Aid only", "material": "Aid, cooperation, de-escalation"}[d])
    window = c3.radio("Window (weeks)", [2, 4, 8], index=1, horizontal=False,
                      help="Support is passed on 'in the same or the next window'. Longer windows are looser: more "
                           "chains qualify, and the order is known less precisely.")
    null = c4.radio("Compare with", ["matched", "shuffle"], horizontal=False,
                    format_func=lambda n: {"matched": "Volume and shared busy weeks", "shuffle": "Volume only"}[n])

    s = res["summary"]
    row = s[(s.definition == definition) & (s.kind == kind) & (s.window_weeks == window) & (s["null"] == null)]
    if row.empty:
        st.info("This combination has not been computed.")
        return
    r = row.iloc[0]
    survivors = int(r["q_le_0.10"])
    st.markdown(
        f"**{int(r.tested):,}** chains with weight 3 or more were tested. **{int(r['p_le_0.01'])}** pass an uncorrected "
        f"1% test, while pure chance would give about **{r.expected_by_chance:.0f}**. After the correction "
        f"**{survivors}** chain{'s' if survivors != 1 else ''} remain.")

    t = res["chains"]
    t = t[(t.definition == definition) & (t.kind == kind) & (t.window_weeks == window) & (t["null"] == null)]
    only_sig = st.toggle("Only chains that stand out (corrected)", value=True)
    shown = t[t.q <= 0.10] if only_sig else t
    if only_sig and shown.empty:
        st.info("No chain stands out in this setting. That is a result too: the big chains here are explained by "
                "volume" + (" and shared busy weeks." if null == "matched" else "."))
    elif len(shown):
        tbl = shown.sort_values(["weight"], ascending=False).head(40)[
            ["chain", "weight", "same_window", "next_window", "expected", "z", "q"]]
        tbl.columns = ["Chain", "Weight", "Same window", "Next window", "Expected by chance", "Z-score", "Corrected p"]
        st.dataframe(tbl.round(3), hide_index=True, width="stretch",
                     height=min(420, 40 + 35 * len(tbl)),
                     column_config={"Weight": st.column_config.NumberColumn(
                         help="Sum over windows of the weaker of the two links. 'Same window': order unknown. "
                              "'Next window': the second link comes in the window after the first.")})

    # sensitivity to the window length and the null model
    sens = s[(s.definition == definition) & (s.kind == kind)].pivot_table(
        index="window_weeks", columns="null", values="q_le_0.10", aggfunc="first")
    sens.columns = [{"matched": "Volume and shared busy weeks", "shuffle": "Volume only"}[c] for c in sens.columns]
    sens.index = [f"{w} weeks" for w in sens.index]
    with st.expander("How sensitive is this to the window length and the comparison?"):
        st.caption("Chains that remain after correction, for each window length and each way of comparing.")
        st.dataframe(sens, width="stretch")
        st.caption("A chain worth taking seriously should appear for more than one window length. "
                   "If the numbers jump around, treat single chains as leads, not findings.")

    # brokers and cycles
    with st.expander("Countries that pass support on, and loops"):
        b = res["brokers"]
        b = b[b["null"] == null].sort_values("score", ascending=False).head(12)
        st.markdown("**Pass-on score** (aid, 4-week windows): for each country the sum over windows of the smaller "
                    "of the aid it receives and the aid it gives. High means it is reported receiving and giving "
                    "in the same period.")
        st.dataframe(b[["country", "score", "expected", "z", "q"]].rename(columns={
            "country": "Country", "score": "Score", "expected": "Expected by chance", "z": "Z-score",
            "q": "Corrected p"}).round(2), hide_index=True, width="stretch")
        cyc = res["cycles"]
        if cyc.empty:
            st.markdown("**Loops** (A → B → C → A within two consecutive windows): none found in the aid graph.")
        else:
            cyc = cyc.assign(q=ch.benjamini_hochberg(cyc.p.values))
            n_sig = int((cyc.q <= 0.10).sum())
            st.markdown(
                f"**Loops** (A → B → C → A within two consecutive windows): **{len(cyc)}** are reported in the aid "
                f"graph. After correcting for the number of loops tested, **{n_sig}** are more frequent than busy "
                f"countries would produce by chance" + (": " if n_sig else "; the rest are what volume alone gives."))
            shown = cyc.sort_values("p").head(10)
            st.dataframe(shown[["cycle", "weight", "expected", "q"]].rename(columns={
                "cycle": "Loop", "weight": "Weight", "expected": "Expected by chance", "q": "Corrected p"}).round(3),
                hide_index=True, width="stretch")


def _words_section(res: dict, names: dict, nodes: list, chains_df: pd.DataFrame) -> None:
    """Section C: do cooperative words come before material support?"""
    st.subheader("C · Do cooperative words come before material support?")
    st.markdown(
        "Words (statements, talks, agreements) are cheap and show up long before aid. We ask: country pairs that "
        "exchanged cooperative words in the first part of the period, do they get a *material* link later more often "
        "than comparable pairs that did not? Each pair with words is compared with the pair that is **most similar in "
        "activity** (how much the actor and the recipient are in the news, and how much the pair is in the news "
        "otherwise), so busy countries are not simply compared with quiet ones. The last third of the period is held "
        "out as the place where we look for the outcome.")
    your_words("chains_words")
    p = res["prediction"]
    c1, c2, c3, c4 = st.columns(4)
    mat = c1.radio("Material support is", ["aid", "material"], format_func=lambda m: {
        "aid": "Aid only", "material": "Aid, cooperation, de-escalation"}[m])
    new_only = c2.radio("Pairs", [True, False], format_func=lambda b: {
        True: "Only pairs without earlier material support (a NEW link)", False: "All pairs (includes existing links)"}[b])
    k = c3.radio("Words needed in the first period", sorted(p.k_verbal.unique()),
                 index=1, format_func=lambda x: f"at least {x} events")
    out = c4.radio("Outcome in the last third", sorted(p.outcome_min.unique()),
                   format_func=lambda x: f"at least {x} event{'s' if x > 1 else ''}")
    row = p[(p.material == mat) & (p.only_new == new_only) & (p.k_verbal == k) & (p.outcome_min == out)]
    r = row.iloc[0] if len(row) else None
    if r is None or not bool(r.ok):
        st.info("Too few pairs for this setting.")
    else:
        fig = go.Figure(go.Bar(
            x=["Pairs with words", "Most similar pairs without words", "All pairs without words"],
            y=[100 * r.rate_treated, 100 * r.rate_matched, 100 * r.rate_all_untreated],
            marker_color=[PRIMARY, ORANGE, GREY],
            text=[f"{100 * v:.0f}%" for v in (r.rate_treated, r.rate_matched, r.rate_all_untreated)],
            textposition="outside"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="% that get material support later",
                          range=[0, max(100 * r.rate_treated, 5) * 1.25]), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        sig = r.ci_low > 0
        if r.max_abs_smd_after > 0.25:
            st.warning(
                f"**Do not read this as an effect.** The matching could not make the pairs comparable: after matching, "
                f"the pairs with words still differ strongly in activity from the pairs they were compared with "
                f"(largest standardised difference {r.max_abs_smd_after:.2f}; up to 0.25 is acceptable). "
                f"Pairs with words are far more active than any pair without words. Choose *new link* above for a "
                f"fairer comparison.")
        else:
            st.markdown(
                f"**{int(r.n_treated)}** country pairs had words. After matching on activity, the gap is "
                f"**{100 * r.difference:+.1f} percentage points** (95% interval {100 * r.ci_low:+.1f} to "
                f"{100 * r.ci_high:+.1f}). "
                + ("The interval stays above zero." if sig else "The interval includes zero, so this setting gives no firm evidence."))
            quality = "very good" if r.max_abs_smd_after <= 0.1 else "acceptable"
            st.caption(f"Balance check: after matching, the pairs differ by at most {r.max_abs_smd_after:.2f} "
                       f"standardised units in activity ({quality}; up to 0.25 is acceptable).")
    with st.expander("All settings"):
        show = p[p.material == mat].copy()
        show["rate_treated"], show["rate_matched"] = 100 * show.rate_treated, 100 * show.rate_matched
        show["difference"], show["ci_low"], show["ci_high"] = 100 * show.difference, 100 * show.ci_low, 100 * show.ci_high
        show["comparable"] = np.where(show.max_abs_smd_after > 0.25, "no: do not use", "yes")
        show = show[["only_new", "k_verbal", "outcome_min", "n_treated", "rate_treated", "rate_matched", "difference",
                     "ci_low", "ci_high", "max_abs_smd_after", "comparable"]]
        show.columns = ["New links only", "Words ≥", "Outcome ≥", "Pairs with words", "% treated", "% matched",
                        "Difference (points)", "CI low", "CI high", "Balance (max)", "Pairs comparable?"]
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        st.caption("Settings with all pairs cannot be compared fairly (pairs with words are much more active than any "
                   "pair without), so only the new-link rows are interpretable. There the gap is clear after 1 event, "
                   "but weak or absent for 3 or more events and for the strictest word threshold, which has few "
                   "pairs. The pattern does not grow steadily with more words, so read it as a modest association, "
                   "not as proof that words lead to aid.")

    # two by two
    st.markdown("##### Words and material support, pair by pair")
    cl = res["classes"]
    cl = cl[cl.material_definition == mat]
    order = [hd.CLASS_BOTH, hd.CLASS_WORDS, hd.CLASS_MATERIAL]
    counts = cl["class"].value_counts().reindex(order).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("Words and material support", f"{counts[hd.CLASS_BOTH]:,}", help="Pairs with ≥3 cooperative-word events and ≥1 material event.")
    c2.metric("Words only", f"{counts[hd.CLASS_WORDS]:,}", help="Candidate latent ties, or empty words.")
    c3.metric("Material support only", f"{counts[hd.CLASS_MATERIAL]:,}", help="Silent support: no cooperative words reported.")
    wo = cl[cl["class"] == hd.CLASS_WORDS].sort_values("verbal", ascending=False).head(15)
    wo = wo.assign(pair=wo.a_id.map(names) + " → " + wo.r_id.map(names))
    st.markdown("**Strongest word-only ties** (many cooperative statements or talks, no material support reported): "
                "either empty words, or support the English-language news does not report. We cannot tell which.")
    st.dataframe(wo[["pair", "verbal", "material"]].rename(columns={"pair": "Pair", "verbal": "Word events",
                 "material": "Material events"}), hide_index=True, width="stretch")

    st.markdown("##### Candidate latent paths")
    st.markdown("A chain A → B → C where A and C exchange cooperative words but no support between them is reported: "
                "support may run through B, and the words hint at a direct tie. A lead to look at, not a finding.")
    t = res["chains"]
    t = t[(t.definition == mat) & (t.kind == "support") & (t.window_weeks == 4) & (t["null"] == "matched")]
    pool = t[t.p <= 0.05]
    label = "chains that pass an uncorrected 5% test (exploratory)"
    if pool.empty:
        pool, label = t.sort_values("weight", ascending=False).head(150), "the 150 heaviest chains (no chain passes a test)"
    lat = hd.candidate_latent_ties(pool, cl.assign(), nodes)
    if lat.empty:
        st.info("No candidate latent path among " + label + ".")
    else:
        st.caption("Chosen from " + label + ".")
        lat = lat.sort_values("ac_verbal", ascending=False).head(15)
        lat["end points"] = lat.a.map(names) + " ⇢ " + lat.c.map(names)
        st.dataframe(lat[["chain", "end points", "weight", "ac_verbal", "ac_material"]].rename(columns={
            "chain": "Chain", "end points": "Words between the end points", "weight": "Chain weight",
            "ac_verbal": "Word events A→C", "ac_material": "Material events A→C"}), hide_index=True, width="stretch")


def render(data: Data) -> None:
    st.title("Step 4 · Support chains")
    files = load_edges()
    if files is None:
        st.error("No event data yet. Run `.venv/bin/python scripts/build_edge_counts.py --run` (free, ~10 minutes).")
        return
    events, edges, names, meta = files["events"], files["edges"], files["names"], files["meta"]
    nodes = list(names)
    res = load_chain_results(chain_stamp())

    your_words("chains_intro")
    if res is not None:
        n_aid = int(((events.event_type == "AID")).sum())
        st.success(_plain_summary(res, len(names), n_aid, meta["latest"]))
    st.info(FRAMING_NOTE)

    # ── A. the map ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("A · Who supports whom, as reported")
    your_words("chains_map")
    c1, c2, c3 = st.columns([2, 3, 2], gap="large")
    layer_label = c1.radio("Show", ["Actions", "Words"], horizontal=True,
                           help="Actions are material events (aid, sanctions, attacks). Words are statements, talks and accusations.")
    layer = "material" if layer_label == "Actions" else "words"
    category = c1.radio("Kind", ["Cooperation", "Conflict"], horizontal=True).lower()
    choices = SELECTIONS[(layer, category)]
    pick = c2.selectbox("Counted as", list(choices))
    months = sorted(pd.to_datetime(events.event_date).dt.strftime("%Y-%m").unique())
    m0, m1 = c3.select_slider("Period (months)", options=months, value=(months[0], months[-1]))
    top = c3.slider("Arcs shown", 20, 250, 80, step=10, help="The heaviest pairs in the period. More arcs, more clutter.")

    until = (pd.Period(m1, "M").end_time).strftime("%Y-%m-%d")
    pairs = ch.pair_weights(events, edges, choices[pick], since=f"{m0}-01", until=until)
    iso = tuple((int(r.country_id), r.iso_alpha3) for r in data.countries.itertuples() if isinstance(r.iso_alpha3, str))
    cent = centroids_by_id(iso)
    focus_name = st.selectbox("Follow one country back and forward (two steps)", ["(none)"] + sorted(names.values()))
    up = down = None
    if focus_name != "(none)":
        fid = next(k for k, v in names.items() if v == focus_name)
        up = ch.reach(pairs, fid, 2, "upstream")
        down = ch.reach(pairs, fid, 2, "downstream")
    color = PRIMARY if category == "cooperation" else ORANGE
    if pairs.empty:
        st.info("No events of this kind in the period.")
    else:
        fig, drawn = _arcs_figure(pairs, cent, names, color, up, down, top)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"{len(pairs):,} country pairs with at least one reported event in the period; the {drawn} heaviest are drawn "
            f"(arc width = number of events; each arc curves to the left of its direction; the big dot is the receiver). "
            + ("Dark blue: reported supporters of the chosen country (up to two steps back); teal: who it is reported "
               "supporting (up to two steps on)." if focus_name != "(none)" else ""))
    if focus_name != "(none)":
        L, R = st.columns(2)
        for col, df, title, a, b in ((L, up, f"Reported to support {focus_name}", "actor", "recipient"),
                                     (R, down, f"{focus_name} is reported to support", "actor", "recipient")):
            col.markdown(f"**{title}**")
            if df is None or df.empty:
                col.caption("None in this period and selection.")
            else:
                t = df.assign(From=df.actor.map(names), To=df.recipient.map(names))
                col.dataframe(t[["step", "From", "To", "weight"]].rename(columns={"step": "Step", "weight": "Events"}),
                              hide_index=True, width="stretch", height=min(300, 40 + 35 * len(t)))

    if res is None:
        st.warning("The chain statistics have not been computed yet. Run "
                   "`.venv/bin/python scripts/build_chain_results.py` (about 15 minutes, free).")
        return
    st.divider()
    _tests_section(res, names, nodes)
    st.divider()
    _words_section(res, names, nodes, res["chains"])
