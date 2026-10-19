"""Tables and labels for the support-cluster page: everything that is not drawing.

Kept apart from the page so it can be tested: naming the blocks, finding brokers, listing candidate latent ties,
and following block membership over time.
"""

import numpy as np
import pandas as pd

import clusters as cl


def config_id(layer: str, category: str, period: str, log: bool, hubs: int, resolution: float = 1.0) -> str:
    """The key under which scripts/build_cluster_results.py stored one clustering."""
    return f"{layer}|{category}|{period}|log{int(log)}|hub{hubs}|res{resolution}"


def block_names(members: pd.DataFrame, names: dict, top: int = 3) -> dict:
    """{cluster: 'Block 1: USA, Ukraine, UK'}: blocks are named after their busiest members. `members` has the
    columns cluster, country_id, strength."""
    out = {}
    for c, g in members[members.cluster >= 0].groupby("cluster"):
        g = g.sort_values("strength", ascending=False)
        out[int(c)] = f"Block {int(c) + 1}: " + ", ".join(names[i] for i in g.country_id.head(top))
    return out


def align_pair(reference: np.ndarray, other: np.ndarray) -> tuple:
    """Give clusters of `other` the label of the reference cluster they overlap most with, so the same group of countries
    has the same colour on two maps. Returns (reference, aligned other)."""
    a, b = cl.align_over_time([reference, other])
    return a, b


def broker_table(counts: np.ndarray, members: pd.DataFrame, nodes: list, names: dict, min_stability: float = 0.7,
                 top: int = 12) -> pd.DataFrame:
    """Countries that tie several blocks together, strongest first.

    `members` is the stored table for one clustering (cluster, stability, participation, participation_z,
    participation_p, outside_share, country_id). 'Stands out' = participation clearly above what a random graph with the
    same number of events per country gives (z >= 2 and p < 0.05). Also lists the two other blocks a country is tied to
    most. Countries that are not stably clustered are left out: their block is not certain, so neither is their bridge.
    """
    idx = {c: i for i, c in enumerate(nodes)}
    m = members[(members.cluster >= 0) & (members.stability >= min_stability) & members.participation.notna()].copy()
    S = counts + counts.T
    membership = np.full(len(nodes), -1)
    for r in members.itertuples():
        membership[idx[r.country_id]] = r.cluster
    rows = []
    for r in m.itertuples():
        i = idx[r.country_id]
        ties = {int(c): S[i, membership == c].sum() for c in np.unique(membership[membership >= 0]) if c != r.cluster}
        other = [f"Block {c + 1}" for c, w in sorted(ties.items(), key=lambda kv: -kv[1])[:2] if w > 0]
        rows.append({"country_id": r.country_id, "Country": names[r.country_id], "Block": f"Block {int(r.cluster) + 1}",
                     "Participation": r.participation, "Z": r.participation_z, "p": r.participation_p,
                     "Ties outside own block": r.outside_share, "Ties mostly to": ", ".join(other),
                     "Stands out": bool(r.participation_z >= 2 and r.participation_p < 0.05)})
    out = pd.DataFrame(rows)
    return out.sort_values("Participation", ascending=False).head(top).reset_index(drop=True) if len(out) else out


def candidate_latent_pairs(words_counts: np.ndarray, mem_words: np.ndarray, mem_actions: np.ndarray,
                           stab_words: np.ndarray, stab_actions: np.ndarray, nodes: list, names: dict,
                           min_stability: float = 0.7, top: int = 15) -> pd.DataFrame:
    """Pairs in the same WORDS block but in different ACTION blocks, both countries stable in both layers: cooperative
    words without shared material support. Either empty words, or support the English-language news does not report.
    Only pairs with at least one cooperative word event between the two themselves are listed, ranked by that number."""
    n = len(nodes)
    i, j = np.triu_indices(n, k=1)
    s = lambda a: np.nan_to_num(a, nan=0.0)   # noqa: E731
    stable = np.minimum.reduce([s(stab_words)[i], s(stab_words)[j], s(stab_actions)[i], s(stab_actions)[j]]) >= min_stability
    clustered = (mem_words[i] >= 0) & (mem_words[j] >= 0) & (mem_actions[i] >= 0) & (mem_actions[j] >= 0)
    keep = stable & clustered & (mem_words[i] == mem_words[j]) & (mem_actions[i] != mem_actions[j])
    i, j = i[keep], j[keep]
    ev = words_counts[i, j] + words_counts[j, i]
    i, j, ev = i[ev > 0], j[ev > 0], ev[ev > 0]           # only pairs with cooperative words between the two themselves
    df = pd.DataFrame({"Pair": [f"{names[nodes[a]]} – {names[nodes[b]]}" for a, b in zip(i, j)], "Word events": ev.astype(int),
                       "Words block": [f"Block {mem_words[a] + 1}" for a in i],
                       "Action blocks": [f"Block {mem_actions[a] + 1} / Block {mem_actions[b] + 1}" for a, b in zip(i, j)]})
    return df.sort_values("Word events", ascending=False).head(top).reset_index(drop=True)


def membership_over_time(members_by_period: dict, nodes: list) -> pd.DataFrame:
    """A table with countries as rows and periods as columns holding the cluster label, where the same group of
    countries keeps the same label from one period to the next. -1 = no ties in that period."""
    periods = list(members_by_period)
    mats = []
    for p in periods:
        m = members_by_period[p].set_index("country_id").reindex(nodes)
        mats.append(m.cluster.fillna(-1).astype(int).to_numpy())
    aligned = cl.align_over_time(mats)
    return pd.DataFrame(np.array(aligned).T, index=nodes, columns=periods)


def timeline_sankey(aligned: pd.DataFrame, stability: pd.DataFrame | None = None, min_stability: float = 0.0) -> tuple:
    """Nodes and links for an alluvial (Sankey) diagram of block membership over periods.

    Nodes are (period, block) pairs; a link counts the countries that are in block X in one period and block Y in the next.
    Countries below `min_stability` in either period are left out (their block is uncertain). Returns (labels of nodes as
    (period, block) tuples, links as (source index, target index, count) tuples).
    """
    periods = list(aligned.columns)
    nodes_idx, links = {}, []

    def node(p, b):
        if (p, b) not in nodes_idx:
            nodes_idx[(p, b)] = len(nodes_idx)
        return nodes_idx[(p, b)]

    for p0, p1 in zip(periods, periods[1:]):
        a, b = aligned[p0], aligned[p1]
        ok = (a >= 0) & (b >= 0)
        if stability is not None and min_stability > 0:
            ok &= (stability[p0].fillna(0) >= min_stability) & (stability[p1].fillna(0) >= min_stability)
        pairs = pd.DataFrame({"a": a[ok], "b": b[ok]}).value_counts()
        for (x, y), n in pairs.items():
            links.append((node(p0, int(x)), node(p1, int(y)), int(n)))
    return list(nodes_idx), links


def block_changes(aligned: pd.DataFrame, stability: pd.DataFrame | None = None, min_stability: float = 0.7) -> pd.DataFrame:
    """For each country: in how many periods did it change block compared with the period before (only counting
    periods in which it is clustered and stable in both)? A rough measure of who wanders between blocks."""
    periods = list(aligned.columns)
    changes = pd.Series(0, index=aligned.index)
    seen = pd.Series(0, index=aligned.index)
    for p0, p1 in zip(periods, periods[1:]):
        ok = (aligned[p0] >= 0) & (aligned[p1] >= 0)
        if stability is not None:
            ok &= (stability[p0].fillna(0) >= min_stability) & (stability[p1].fillna(0) >= min_stability)
        seen += ok.astype(int)
        changes += (ok & (aligned[p0] != aligned[p1])).astype(int)
    out = pd.DataFrame({"changes": changes, "comparisons": seen})
    out["share"] = np.where(out.comparisons > 0, out.changes / out.comparisons.where(out.comparisons > 0), np.nan)
    return out


def strip_labels(aligned: pd.DataFrame, stability: pd.DataFrame, ids: list, names: dict, min_stability: float = 0.7) -> pd.DataFrame:
    """Text for a countries-by-periods strip: the group number, '?' when the block is not certain enough, '–' when the
    country has no ties in that period."""
    a, s = aligned.loc[ids], stability.loc[ids].fillna(0)
    text = pd.DataFrame(index=a.index, columns=a.columns, dtype=object)
    for col in a.columns:
        text[col] = np.where(a[col] < 0, "–", np.where(s[col] >= min_stability, (a[col] + 1).astype(str), "?"))
    text.index = [names[i] for i in ids]
    return text
