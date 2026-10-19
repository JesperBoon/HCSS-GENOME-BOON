"""Step 4: case studies. For a party in a conflict: which blocks it belongs to in words and in actions, who bridges to the
other blocks, how that moved over time, who supplies it (directly or through others), and the events behind each path.

The blocks come from scripts/build_cluster_results.py. The events behind a path come from the free first page of the API.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import chains as ch
import cluster_tables as ct
import aid_types
import narrative
from views.common import (FRAMING_NOTE, PRIMARY, Data, chain_stamp, cluster_stamp, live_events, load_chain_results,
                          load_cluster_results, load_edges, marked_events, pair_counts, style_strip, your_words)

ORANGE = "#E67E22"
GREY = "#8A97AD"
STABLE = 0.7

# Case studies. `parties` are the countries we follow; `why` states the data fact behind the choice.
CASES = {
    "Russia–Ukraine: who supplies Ukraine, and through whom": {
        "kind": "supply", "parties": ["Ukraine", "Russia"], "key": "case_ukraine",
        "why": "Ukraine is the country that receives the most reported aid in the whole data, and Russia is the fourth. "
               "Both have many supporters, so the question 'directly or through someone else?' matters."},
    "Iran–Israel: the networks around two adversaries": {
        "kind": "supply", "parties": ["Iran", "Israel"], "key": "case_iran",
        "why": "Iran is a hub of reported attacks and Israel is the origin of by far the most reported attacks, so the "
               "supply networks around both are worth a closer look."},
    "Israel–Palestine: aid and the party in between": {
        "kind": "supply", "parties": ["Palestine", "Israel"], "key": "case_palestine",
        "why": "Palestine is the second-biggest receiver of reported aid, and the heaviest aid chain in the data runs "
               "USA → Israel → Palestine."},
}


def _members(res: dict, layer: str, nodes: list) -> pd.DataFrame:
    """Whole-period blocks for one layer (cooperation, event counts, nobody left out), one row per country."""
    cid = ct.config_id(layer, "cooperation", "all", False, 0)
    return res["members"][res["members"].cid == cid].set_index("country_id").reindex(nodes)


def _mismatch_case(res: dict, names: dict, nodes: list):
    """The case chosen by the data: the two countries whose words and actions disagree most, among countries with a certain
    block in both layers. Returns (title, case dict) or None."""
    mw, ma = _members(res, "words", nodes), _members(res, "support", nodes)
    dev = res["deviation"]
    dev = dev[(dev.actions == "support") & (dev.period == "all") & (~dev["log"]) & (dev.hubs == 0)].set_index("country_id").deviation
    ok = [c for c in nodes if np.nan_to_num(mw.stability.get(c, np.nan)) >= STABLE and np.nan_to_num(ma.stability.get(c, np.nan)) >= STABLE]
    top = dev.reindex(ok).dropna().sort_values(ascending=False).head(2)
    if len(top) < 2:
        return None
    a, b = [names[c] for c in top.index]
    why = (f"{a} and {b} are the two countries whose words and actions disagree most. Among countries with a certain block in both "
           f"layers, {100 * top.iloc[0]:.0f}% and {100 * top.iloc[1]:.0f}% of their partners in one layer are not their partners in "
           f"the other (average of the two directions). Their cooperative words put them in a different block from their aid.")
    return f"Words and actions apart: {a} and {b}", {"kind": "mismatch", "parties": [a, b], "key": "case_mismatch", "why": why}


def _block_context(res: dict, files: dict, party: int, party_name: str, names: dict, nodes: list, meta_clusters: dict):
    """Section 1: the blocks around the party. Returns the set of countries in its action block (or None)."""
    st.subheader(f"1 · The blocks around {party_name}")
    mw, ma = _members(res, "words", nodes), _members(res, "support", nodes)
    cols = st.columns(2, gap="large")
    block_ids = None
    for col, lbl, m in ((cols[0], "Actions", ma), (cols[1], "Words", mw)):
        with col:
            if party not in m.index or not m.cluster.notna().get(party, False) or m.cluster[party] < 0:
                st.markdown(f"**{lbl}.** {party_name} has no ties here.")
                continue
            k, stab = int(m.cluster[party]), m.stability[party]
            mates = m[m.cluster == k].sort_values("strength", ascending=False)
            if lbl == "Actions":
                block_ids = set(mates.index)
            line = ", ".join(names[c] + ("*" if not np.nan_to_num(m.stability[c]) >= STABLE else "") for c in mates.index[:16])
            st.markdown(f"**{lbl}.** {party_name} is in block {k + 1} (certainty {stab:.2f}{'' if stab >= STABLE else ', uncertain'}) "
                        f"with {len(mates)} countries: {line}{' …' if len(mates) > 16 else ''}")
    st.caption("A * after a country means its block is uncertain.")

    st.markdown("**Who bridges to the other blocks?** Countries in the same action block as the party with the most ties to other blocks:")
    W = pair_counts("support", "cooperation", meta_clusters["since"], meta_clusters["latest"])
    bt = ct.broker_table(W, ma.dropna(subset=["participation"]).reset_index(), nodes, names, STABLE, top=50)
    bridges = pd.DataFrame()
    if len(bt) and party in ma.index and ma.cluster.get(party, -1) >= 0:
        bridges = bt[(bt.Block == f"Block {int(ma.cluster[party]) + 1}") & (bt.country_id != party)].head(4)
    if bridges.empty:
        st.caption("No country in this block has a certain place and a bridge score.")
    else:
        t = bridges[["Country", "Participation", "Ties outside own block", "Ties mostly to", "Stands out"]].copy()
        t["Ties outside own block"] = 100 * t["Ties outside own block"]
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "Participation": st.column_config.NumberColumn("Bridge score", format="%.2f"),
            "Ties outside own block": st.column_config.ProgressColumn("Ties outside own block", min_value=0, max_value=100, format="%.0f%%")})

    st.markdown("**How did this move over time?** Quarter by quarter (number = group, ? = uncertain, – = no ties):")
    quarters = list(pd.period_range(meta_clusters["since"], meta_clusters["latest"], freq="Q").astype(str))
    follow = [party] + (list(bridges.country_id.head(3)) if len(bridges) else [])
    for lbl, layer in (("Words", "words"), ("Actions", "support")):
        mbp, stabs = {}, {}
        for q in quarters:
            m = res["members"][res["members"].cid == ct.config_id(layer, "cooperation", q, False, 0)]
            if len(m):
                mbp[q] = m
                stabs[q] = m.set_index("country_id").stability.reindex(nodes)
        aligned = ct.membership_over_time(mbp, nodes)
        text = ct.strip_labels(aligned, pd.DataFrame(stabs), follow, names, STABLE)
        st.caption(lbl)
        st.dataframe(style_strip(text), width="stretch")
    return block_ids


def _sankey(pairs: pd.DataFrame, party: int, names: dict, n_suppliers: int = 12, n_middle: int = 8):
    """Suppliers -> intermediaries -> the party, from the aid pairs of one period.

    Intermediary: a country that is reported giving aid to the party and also receiving aid from another country
    in the same period. Everyone else who gives to the party, or to an intermediary, is a supplier.
    """
    to_party = pairs[pairs.recipient == party]
    candidates = set(to_party.actor) - {party}
    received = pairs[pairs.recipient.isin(candidates) & (pairs.actor != party)]
    middle_ids = set(received.recipient)
    middle = (to_party[to_party.actor.isin(middle_ids)].sort_values("weight", ascending=False)
              .head(n_middle).actor.tolist())
    mset = set(middle)
    direct = to_party[~to_party.actor.isin(mset)]
    s_to_m = received[received.recipient.isin(mset) & ~received.actor.isin(mset)]
    flows = pd.concat([
        s_to_m.rename(columns={"actor": "src", "recipient": "dst"}).assign(kind="supplier → intermediary"),
        to_party[to_party.actor.isin(mset)].rename(columns={"actor": "src", "recipient": "dst"}).assign(kind="intermediary → party"),
        direct.rename(columns={"actor": "src", "recipient": "dst"}).assign(kind="supplier → party (direct)")],
        ignore_index=True)
    if flows.empty:
        return None, flows
    top_sup = (flows[flows.kind != "intermediary → party"].groupby("src").weight.sum()
               .sort_values(ascending=False).head(n_suppliers).index)
    flows = flows[(flows.kind == "intermediary → party") | flows.src.isin(top_sup)]
    suppliers = [c for c in top_sup if c not in mset]
    order = suppliers + middle + [party]
    pos = {c: i for i, c in enumerate(order)}
    colors = [GREY] * len(suppliers) + [ORANGE] * len(middle) + [PRIMARY]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[names[c] for c in order], color=colors, pad=14, thickness=16),
        link=dict(source=[pos[s] for s in flows.src], target=[pos[d] for d in flows.dst],
                  value=flows.weight.tolist(), label=flows.kind.tolist(),
                  color=["rgba(138,151,173,0.35)" if k == "supplier → party (direct)" else "rgba(230,126,34,0.35)"
                         for k in flows.kind])))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    return fig, flows


def _events_table(legs: list, event_type: str, names: dict, since: str, until: str, support_only: bool = False):
    """The real events behind a path (free first page of the API), as a table; also warns about the 250 limit.
    With support_only, AID events that are exchanges, evacuations, trade measures or ceremonies are left out."""
    rows, over = [], False
    for a, b in legs:
        evs, total = live_events(a, b, event_type, since, until)
        over = over or total > 250
        if support_only:
            evs = [e for e in evs if aid_types.is_support(aid_types.classify_aid(e["event_summary"], e["core_sentence"]))]
        for e in evs:
            rows.append({"Date": e["event_date"], "Link": f"{names[a]} → {names[b]}", "Intensity": e["intensity"],
                         "Event": e["core_sentence"], "Quote": (e["source_quote"] or "")[:260],
                         "Reports": e["article_count"], "Source": (e.get("article_links") or [""])[0],
                         "event_type": e["event_type"], "core_sentence": e["core_sentence"], "event_date": e["event_date"],
                         "intensity": e["intensity"], "source_quote": e["source_quote"]})
    if over:
        st.warning("More than 250 events in one leg: only the newest 250 are shown.")
    if not rows:
        st.info("No events found for this path in the chosen period.")
        return None
    tl = pd.DataFrame(rows).sort_values("Date", ascending=False)
    st.dataframe(tl[["Date", "Link", "Intensity", "Event", "Quote", "Reports", "Source"]], hide_index=True,
                 width="stretch", height=380,
                 column_config={"Source": st.column_config.LinkColumn("Source", display_text="open"),
                                "Quote": st.column_config.TextColumn("Quote", width="large")})
    st.caption(f"{len(tl)} events from the free first page of the GENOME API, newest first.")
    return tl


def _narrative(label: str, window: str, tl: pd.DataFrame, key: str) -> None:
    """Optional short summary by Claude, behind a switch; the dashboard works without it."""
    st.markdown("##### Optional: a short summary written by Claude")
    st.caption("Off by default. It sends only the events shown above to Anthropic's API and needs an "
               "`ANTHROPIC_API_KEY` environment variable. The dashboard works without it.")
    if st.toggle("Write a short summary", key=f"nar_{key}"):
        if st.button("Write summary", key=f"btn_{key}"):
            try:
                st.session_state[f"sum_{key}"] = (label, narrative.summarize(label, window, tl.to_dict("records")))
            except narrative.MissingKey as exc:
                st.warning(str(exc))
            except Exception as exc:                        # network or API errors must not break the page
                st.error(f"The summary could not be written: {type(exc).__name__}")
        saved = st.session_state.get(f"sum_{key}")
        if saved and saved[0] == label:
            st.markdown(f"> {saved[1]}")
            st.caption("Written by an AI from the events above. It may contain mistakes; check it against the events.")


def _supply_view(files: dict, party: int, party_name: str, names: dict, block_ids) -> None:
    """Section 2: who supplies the party, directly or through others, with a time slider."""
    events, edges = files["events"], files["edges"]
    st.subheader(f"2 · Who supplies {party_name}, and through whom")
    aid_events = events[events.event_type == "AID"]
    days = pd.to_datetime(aid_events.event_date)
    first, last = days.min().normalize(), days.max().normalize()
    c1, c2, c3 = st.columns([1, 3, 2], gap="large")
    weeks = c1.select_slider("Period length", options=[4, 8, 13, 26, 52], value=13, format_func=lambda w: f"{w} weeks")
    starts = [d.strftime("%Y-%m-%d") for d in pd.date_range(first, last - pd.Timedelta(weeks=weeks), freq="2W-MON")]
    since = c2.select_slider("Period starts", options=starts, value=starts[len(starts) // 2])
    until = (pd.Timestamp(since) + pd.Timedelta(weeks=weeks) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    inside = c3.toggle("Only inside the party's action block", value=False, disabled=block_ids is None,
                       help="Keeps only aid between countries that are in the same block of actions as the party: the chains inside that block.")
    support_only = c3.toggle("Support only", value=True,
                             help="Leaves out AID that is an exchange of prisoners or bodies, an evacuation, a trade measure or a ceremony.")
    st.caption(f"Reported aid events from {since} to {until}. Move the start to see how the picture changes.")

    pairs = ch.pair_weights(marked_events() if support_only else events, edges, ["AID_SUPPORT"] if support_only else ["AID"],
                            since=since, until=until)
    if inside and block_ids:
        pairs = pairs[pairs.actor.isin(block_ids) & pairs.recipient.isin(block_ids)]
    fig, flows = _sankey(pairs, party, names)
    if fig is None:
        st.info(f"No aid to {party_name} is reported in this period. Try another period" + (" or switch off the block filter." if inside else "."))
        return
    st.plotly_chart(fig, width="stretch")
    st.caption("Left: countries reported giving aid to the party or to an intermediary. Middle: intermediaries, countries reported both "
               "receiving aid and giving aid to the party in this period. Right: the party. Band width = number of reported aid events. "
               "At most the 12 biggest suppliers and 8 intermediaries are shown. An intermediary is a country that appears on both sides "
               "in the same period; that is a **reported pattern**, it does not mean the aid it gave came from what it received. Big "
               "donors such as the USA often show up as intermediaries simply because they are also reported receiving some aid "
               "(relief after a disaster, for example).")

    st.markdown("##### The events behind a path")
    st.warning("**Read the events.** In GENOME the class AID is wide: it also holds humanitarian exchanges and military help. "
               "Russia returning the bodies of soldiers to Ukraine, for example, is coded 'Russia AID Ukraine', and North Korea "
               "sending troops to Russia is coded 'AID' too. A path built from such events is a reported pattern, not necessarily "
               "support in the usual sense.")
    two = flows[flows.kind == "intermediary → party"][["src", "weight"]].rename(columns={"src": "mid", "weight": "w2"})
    first_leg = flows[flows.kind == "supplier → intermediary"].rename(columns={"src": "sup", "dst": "mid", "weight": "w1"})
    paths = first_leg.merge(two, on="mid")
    if paths.empty:
        st.info("No two-step path in this period: all reported aid to the party comes directly or through intermediaries without a "
                "reported supplier. Choose a longer period.")
        return
    paths["strength"] = paths[["w1", "w2"]].min(axis=1)
    paths = paths.sort_values("strength", ascending=False).head(15).reset_index(drop=True)
    labels = [f"{names[r.sup]} → {names[r.mid]} → {party_name}  (weakest link: {int(r.strength)} events)" for r in paths.itertuples()]
    chosen = st.selectbox("Path", labels)
    r = paths.iloc[labels.index(chosen)]

    res = load_chain_results(chain_stamp())
    if res is not None:
        t = res["chains"]
        t = t[(t.definition == "aid") & (t.kind == "support") & (t.window_weeks == 4) & (t["null"] == "matched")
              & (t.a == r.sup) & (t.b == r.mid) & (t.c == party)]
        if len(t):
            q = t.iloc[0]
            verdict = ("stands out from what volume and shared busy weeks would give" if q.q <= 0.10
                       else "does **not** stand out from what volume and shared busy weeks would give")
            st.markdown(f"Over the whole period (4-week windows) this chain has weight **{q.weight:.0f}** and {verdict} "
                        f"(corrected p = {q.q:.2f}; {q.expected:.0f} expected by chance).")
        else:
            st.caption("This chain is too light (weight under 3 over the whole period) to be in the statistics.")
    tl = _events_table([(int(r.sup), int(r.mid)), (int(r.mid), party)], "AID", names, since, until, support_only)
    if tl is not None:
        _narrative(chosen.split("  (")[0], f"{since} to {until}", tl, "supply")


def _mismatch_view(files: dict, res: dict, focus: int, focus_name: str, names: dict, nodes: list, meta_clusters: dict) -> None:
    """Section 2 for the mismatch case: who the country talks cooperatively with, against who it acts with."""
    st.subheader(f"2 · Words and actions of {focus_name}")
    since, until = meta_clusters["since"], meta_clusters["latest"]
    Ww = pair_counts("words", "cooperation", since, until)
    Wm = pair_counts("support", "cooperation", since, until)
    i = nodes.index(focus)
    fig = go.Figure()
    for lbl, M, colour in (("Cooperative words", Ww, PRIMARY), ("Material cooperation (actions)", Wm, ORANGE)):
        v = M[i] + M[:, i]
        top = np.argsort(-v)[:8]
        fig.add_trace(go.Bar(y=[names[nodes[j]] for j in top][::-1], x=[v[j] for j in top][::-1], orientation="h", name=lbl,
                             marker_color=colour, visible=True))
    fig.update_layout(barmode="group", height=420, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Reported events with the country",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The eight countries {focus_name} has the most cooperative words with, and the eight it has the most material cooperation "
               "with (given plus received). Compare the two lists: that is the disagreement.")
    top_action = int(np.argmax(Wm[i] + Wm[:, i]))
    partner = nodes[top_action]
    marked = marked_events().merge(files["edges"], on="event_id")
    both_ways = pd.concat([marked.query("actor_id == @focus and recipient_id == @partner"),
                           marked.query("actor_id == @partner and recipient_id == @focus")])
    both_ways = both_ways[both_ways.event_type == "AID_SUPPORT"]
    st.markdown(f"##### The events behind the strongest action tie: {focus_name} and {names[partner]}")
    if both_ways.empty:
        st.info("No material cooperation events between the two.")
        return
    et = "AID"                                               # the API type; the events are filtered to support below
    direction = (focus, partner) if len(both_ways[both_ways.actor_id == focus]) >= len(both_ways[both_ways.actor_id == partner]) else (partner, focus)
    st.caption(f"Support aid only (military, humanitarian, financial). Showing {names[direction[0]]} → {names[direction[1]]}.")
    tl = _events_table([direction], et, names, since, until, support_only=True)
    if tl is not None:
        _narrative(f"{names[direction[0]]} → {names[direction[1]]} ({et})", f"{since} to {until}", tl, "mismatch")


def render(data: Data) -> None:
    st.title("Step 4 · Case studies")
    files = load_edges()
    if files is None:
        st.error("No event data yet. Run `.venv/bin/python scripts/build_edge_counts.py --run` (free).")
        return
    names = files["names"]
    nodes = list(names)
    ids = {v: k for k, v in names.items()}
    res = load_cluster_results(cluster_stamp())
    st.info(FRAMING_NOTE + " A block is a statistical pattern in reported events, not proof of an alliance.")
    your_words("case_intro")

    cases = dict(CASES)
    if res is not None:
        extra = _mismatch_case(res, names, nodes)
        if extra:
            cases[extra[0]] = extra[1]
    case_name = st.selectbox("Case", list(cases))
    case = cases[case_name]
    st.markdown(f"**Why this case.** {case['why']}")
    your_words(case["key"])
    label = "Focus on" if case["kind"] == "mismatch" else "Follow the supporters of"
    party_name = st.radio(label, case["parties"], horizontal=True)
    party = ids[party_name]

    block_ids = None
    if res is not None:
        block_ids = _block_context(res, files, party, party_name, names, nodes, res["meta"])
    else:
        st.warning("The cluster results are not built yet (scripts/build_cluster_results.py), so the blocks are not shown.")
    if case["kind"] == "supply":
        _supply_view(files, party, party_name, names, block_ids)
    elif res is not None:
        _mismatch_view(files, res, party, party_name, names, nodes, res["meta"])
