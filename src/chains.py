"""Support chains: who passes material support on to whom, and is that more than volume alone?

What this measures, and what it does not
  GENOME reports events: "A AID B" on a date. It has no amounts (no money, no weapons) and its intensity is
  ordinal. A chain A -> B -> C below means: in the news, A is reported giving material support to B and B is
  reported giving material support to C, in the same or the next time window. It is a reported, time-ordered
  *path* of events, not a proven flow. Time closeness is not causation. "Hidden" means "not observed in this
  English-language newswire corpus", not "secret".

Data shape
  A Graph holds a dense array counts[a, r, t]: the number of events of one layer/category with country a as actor
  and country r as recipient in time window t. Windows are `window_weeks` weeks wide (>= 1), so order is known
  only at the level of windows: A -> B in window t and B -> C in window t is "same window" (order unknown),
  and A -> B in t and B -> C in t + 1 is "next window" (ordered).

Null model
  Big countries make chains through volume alone. For every edge (a, r) we shuffle its counts over the time
  windows at random (its total stays the same, the timing is destroyed) and recompute everything. A chain is
  only interesting if its observed weight is larger than what such shuffles produce.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

import taxonomy as tx


@dataclass
class Graph:
    nodes: list           # country ids, row/column order of `counts`
    start: date           # first day of window 0 (a Monday)
    window_weeks: int
    counts: np.ndarray    # shape (n, n, T), float


# ── building a graph from the stored edge files ──────────────────────────────

def n_windows(start: date, end: date, window_weeks: int) -> int:
    return ((end - start).days // (7 * window_weeks)) + 1


def build_graph(events: pd.DataFrame, edges: pd.DataFrame, leaves: pd.DataFrame, nodes: list, start: date,
                end: date, window_weeks: int, layer: str, category: str, event_types: list | None = None) -> Graph:
    """counts[a, r, t] for one layer ('material' or 'words') and category ('cooperation' or 'conflict').

    Events of closed windows come with their exact date. The few windows that held more than 250 events
    exist only as a count over their window; those counts are spread evenly over the days of the window.
    """
    idx = {c: i for i, c in enumerate(nodes)}
    T = n_windows(start, end, window_weeks)
    counts = np.zeros((len(nodes), len(nodes), T))
    types = event_types or [t for t in tx.ALL_TYPES if tx.layer_of(t) == layer and tx.category_of(t) == category]

    ev = events[events.event_type.isin(types)].copy()
    ev["event_date"] = pd.to_datetime(ev.event_date)
    ev["w"] = ((ev.event_date - pd.Timestamp(start)).dt.days // (7 * window_weeks)).astype(int)
    m = edges.merge(ev[["event_id", "w"]], on="event_id")
    m = m[(m.w >= 0) & (m.w < T) & m.actor_id.isin(idx) & m.recipient_id.isin(idx)]
    g = m.groupby(["actor_id", "recipient_id", "w"]).size()
    for (a, r, w), n in g.items():
        counts[idx[a], idx[r], w] += n

    lc = leaves[(leaves.source == "count") & leaves.event_type.isin(types)]
    for row in lc.itertuples():
        if row.actor_id not in idx or row.recipient_id not in idx:
            continue
        days = pd.date_range(row.window_start, row.window_end)
        for d in days:
            w = (d.date() - start).days // (7 * window_weeks)
            if 0 <= w < T:
                counts[idx[row.actor_id], idx[row.recipient_id], w] += row.count / len(days)
    return Graph(list(nodes), start, window_weeks, counts)


# ── chains of length two ─────────────────────────────────────────────────────

def chain_weights(X: np.ndarray, Y: np.ndarray, combine: str = "min"):
    """Weight of every chain A -> B -> C, from edge series X (A->B) and Y (B->C).

    Returns (same, next), both of shape (n, n, n) indexed [A, B, C]:
      same  = sum over t of f(X[A,B,t], Y[B,C,t])        B passes on in the same window (order unknown)
      next  = sum over t of f(X[A,B,t], Y[B,C,t+1])      B passes on in the following window (ordered)
    f is min (conservative: the chain is only as strong as its weakest link) or the product.
    Chains that return to A (A == C) are removed.
    """
    f = {"min": np.minimum, "product": np.multiply}[combine]
    x = X[:, :, None, :]                     # A, B, -, T
    y = Y[None, :, :, :]                     # -, B, C, T
    same = f(x, y).sum(axis=3)
    nxt = f(x[..., :-1], y[..., 1:]).sum(axis=3) if X.shape[2] > 1 else np.zeros_like(same)
    n = X.shape[0]
    not_back = ~np.eye(n, dtype=bool)[:, None, :]           # A != C, broadcast over B
    return same * not_back, nxt * not_back


def activity_order(graphs: list) -> np.ndarray:
    """The time windows sorted from quietest to busiest (overall events in the window; with several graphs the
    activity ranks are added, so a week that is busy in both counts as busy)."""
    rank_sum = 0
    for g in graphs:
        per_window = g.reshape(-1, g.shape[-1]).sum(axis=0)
        rank_sum = rank_sum + np.argsort(np.argsort(per_window, kind="stable"), kind="stable")
    return np.argsort(rank_sum, kind="stable")


def shuffle_edges(counts: np.ndarray, rng: np.random.Generator, order: np.ndarray | None = None,
                  tolerance: float = 3.0) -> np.ndarray:
    """Null model: move the timing of every edge on its own, keeping its total and its pattern of busy and
    quiet windows.

    Without `order` any window can swap with any other ("shuffle").
    With `order` (the windows from quiet to busy, see `activity_order`) a window's counts can only move to a
    window of similar overall activity: each window's position in the activity ranking is nudged by random
    noise of about `tolerance` places and the windows are re-sorted. A week that is busy for everyone then
    stays among the busy weeks, so two edges that are both busy in a war week do not look coordinated because
    of it ("matched"). tolerance -> 0 changes nothing, tolerance -> infinity is the plain shuffle.
    Works on the last axis, so it serves full arrays (n, n, T) and row blocks (E, T) alike.
    """
    if order is None:
        return np.take_along_axis(counts, rng.random(counts.shape).argsort(axis=-1), axis=-1)
    by_rank = counts[..., order]
    keys = np.arange(counts.shape[-1]) + tolerance * rng.standard_normal(by_rank.shape)
    moved = np.take_along_axis(by_rank, keys.argsort(axis=-1), axis=-1)
    out = np.empty_like(counts)
    out[..., order] = moved
    return out


def _tol(tolerance: float | None, n_windows: int) -> float:
    """Default tolerance: 8% of the number of windows (about 3 places for 37 windows), so the matching is
    equally tight whatever the window length."""
    return 0.08 * n_windows if tolerance is None else tolerance


def _order_for(null: str, graphs: list):
    if null == "shuffle":
        return None
    if null == "matched":
        return activity_order(graphs)
    raise ValueError("null must be 'shuffle' or 'matched'")


def chain_significance(X: np.ndarray, Y: np.ndarray, n_perm: int = 1000, seed: int = 0,
                       combine: str = "min", min_weight: float = 1.0, null: str = "matched",
                       tolerance: float | None = None) -> dict:
    """Observed chain weights against a null model.

    null: "matched" (default: each edge only moves to windows of similar overall activity, so weeks that are
    busy for everyone do not count as coordination) or "shuffle" (any window swaps with any).
    X and Y may be the same array (then both are drawn identically, as one graph). Returns arrays [A, B, C]:
    same, next, weight (= same + next), expected (null mean), z, p (empirical, one-sided:
    share of shuffles with a weight at least as large, counting the observed one).
    """
    rng = np.random.default_rng(seed)
    same, nxt = chain_weights(X, Y, combine)
    obs = same + nxt
    same_graph = X is Y
    order = _order_for(null, [X] if same_graph else [X, Y])
    tolerance = _tol(tolerance, X.shape[-1])
    total = np.zeros_like(obs)
    sq = np.zeros_like(obs)
    ge = np.zeros(obs.shape, dtype=np.int32)
    for _ in range(n_perm):
        Xp = shuffle_edges(X, rng, order, tolerance)
        Yp = Xp if same_graph else shuffle_edges(Y, rng, order, tolerance)
        s, nx = chain_weights(Xp, Yp, combine)
        w = s + nx
        total += w
        sq += w * w
        ge += (w >= obs)
    mean = total / n_perm
    std = np.sqrt(np.maximum(sq / n_perm - mean ** 2, 0))
    z = np.where(std > 0, (obs - mean) / np.where(std > 0, std, 1), np.where(obs > mean, np.inf, 0.0))
    p = (1 + ge) / (1 + n_perm)
    p = np.where(obs >= min_weight, p, 1.0)
    return {"same": same, "next": nxt, "weight": obs, "expected": mean, "z": z, "p": p}


def chains_table(result: dict, nodes: list, names: dict, min_weight: float = 1.0, p_max: float | None = None,
                 top: int | None = None) -> pd.DataFrame:
    """The chains above min_weight (and below p_max, if given) as a table, strongest first."""
    A, B, C = np.nonzero(result["weight"] >= min_weight)
    df = pd.DataFrame({
        "a": [nodes[i] for i in A], "b": [nodes[i] for i in B], "c": [nodes[i] for i in C],
        "weight": result["weight"][A, B, C], "same_window": result["same"][A, B, C],
        "next_window": result["next"][A, B, C], "expected": result["expected"][A, B, C],
        "z": result["z"][A, B, C], "p": result["p"][A, B, C]})
    df["chain"] = df.a.map(names) + " → " + df.b.map(names) + " → " + df.c.map(names)
    if p_max is not None:
        df = df[df.p <= p_max]
    df = df.sort_values(["weight", "z"], ascending=False)
    return df.head(top) if top else df


# ── brokers ──────────────────────────────────────────────────────────────────

def broker_scores(X: np.ndarray) -> np.ndarray:
    """Pass-on score per country: summed over windows, min(support received, support given).

    A country that only gives, or only receives, scores 0. Large when it both receives and passes on
    within the same window.
    """
    received = X.sum(axis=0)                  # [B, T]: events with B as recipient
    given = X.sum(axis=1)                     # [B, T]: events with B as actor
    return np.minimum(received, given).sum(axis=1)


def broker_significance(X: np.ndarray, n_perm: int = 1000, seed: int = 0, null: str = "matched",
                        tolerance: float | None = None) -> dict:
    """Pass-on scores against a null model: observed, expected, z, p."""
    rng = np.random.default_rng(seed)
    order = _order_for(null, [X])
    tolerance = _tol(tolerance, X.shape[-1])
    obs = broker_scores(X)
    total, sq, ge = np.zeros_like(obs), np.zeros_like(obs), np.zeros(obs.shape, dtype=np.int32)
    for _ in range(n_perm):
        w = broker_scores(shuffle_edges(X, rng, order, tolerance))
        total += w
        sq += w * w
        ge += (w >= obs)
    mean = total / n_perm
    std = np.sqrt(np.maximum(sq / n_perm - mean ** 2, 0))
    z = np.where(std > 0, (obs - mean) / np.where(std > 0, std, 1), 0.0)
    return {"score": obs, "expected": mean, "z": z, "p": (1 + ge) / (1 + n_perm)}


# ── cycles ───────────────────────────────────────────────────────────────────

def cycle_weights(X: np.ndarray) -> np.ndarray:
    """Weight of cycles A -> B -> C -> A: in each pair of consecutive windows (t, t+1) pooled together,
    min of the three edges, summed over t. Returned as [A, B, C] with A the smallest index of the cycle,
    so every cycle appears once (in the direction A -> B -> C)."""
    T = X.shape[2]
    pooled = X[..., :-1] + X[..., 1:] if T > 1 else X
    ab = pooled[:, :, None, :]                    # A -> B
    bc = pooled[None, :, :, :]                    # B -> C
    ca = pooled.transpose(1, 0, 2)[:, None, :, :]   # C -> A, indexed [A, -, C]
    w = np.minimum(np.minimum(ab, bc), ca).sum(axis=3)
    n = X.shape[0]
    i = np.arange(n)
    canonical = (i[:, None, None] < i[None, :, None]) & (i[:, None, None] < i[None, None, :]) \
        & (i[None, :, None] != i[None, None, :])
    return w * canonical


def cycle_significance(X: np.ndarray, n_perm: int = 500, seed: int = 0, null: str = "matched",
                       tolerance: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    order = _order_for(null, [X])
    tolerance = _tol(tolerance, X.shape[-1])
    obs = cycle_weights(X)
    ge = np.zeros(obs.shape, dtype=np.int32)
    total = np.zeros_like(obs)
    for _ in range(n_perm):
        w = cycle_weights(shuffle_edges(X, rng, order, tolerance))
        total += w
        ge += (w >= obs)
    return {"weight": obs, "expected": total / n_perm, "p": (1 + ge) / (1 + n_perm)}


# ── many tests at once ───────────────────────────────────────────────────────

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values). With thousands of chains tested, a few hundred
    would pass p < 0.05 by luck alone; q <= 0.1 means that at most about 10% of the chains called
    significant are expected to be chance findings."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def refine_p_values(X: np.ndarray, Y: np.ndarray, triples: np.ndarray, n_perm: int = 20000, seed: int = 0,
                    combine: str = "min", null: str = "matched", tolerance: float | None = None) -> np.ndarray:
    """Empirical p-values with many more shuffles, for a chosen list of chains only.

    `triples` is an array of (A, B, C) index rows. Only the edges those chains use are shuffled, so 20,000
    shuffles take seconds and p-values can go down to 1 / 20,001, which the full-tensor test cannot reach.
    """
    rng = np.random.default_rng(seed)
    f = {"min": np.minimum, "product": np.multiply}[combine]
    order = _order_for(null, [X] if X is Y else [X, Y])
    tolerance = _tol(tolerance, X.shape[-1])
    triples = np.asarray(triples)
    same_graph = X is Y
    ab = sorted({(a, b) for a, b, _ in triples})
    bc = sorted({(b, c) for _, b, c in triples})
    if same_graph:
        rows = sorted(set(ab) | set(bc))
        pos = {e: i for i, e in enumerate(rows)}
        Sx = np.array([X[a, b] for a, b in rows])
        ia = np.array([pos[(a, b)] for a, b, _ in triples])
        ib = np.array([pos[(b, c)] for _, b, c in triples])
        Sy = None
    else:
        posx, posy = {e: i for i, e in enumerate(ab)}, {e: i for i, e in enumerate(bc)}
        Sx = np.array([X[a, b] for a, b in ab])
        Sy = np.array([Y[b, c] for b, c in bc])
        ia = np.array([posx[(a, b)] for a, b, _ in triples])
        ib = np.array([posy[(b, c)] for _, b, c in triples])

    def weight(left, right):
        return f(left[:, :], right[:, :]).sum(axis=1) + f(left[:, :-1], right[:, 1:]).sum(axis=1)

    obs = weight(Sx[ia], (Sx if same_graph else Sy)[ib])
    ge = np.zeros(len(triples), dtype=np.int64)
    for _ in range(n_perm):
        Sxp = shuffle_edges(Sx, rng, order, tolerance)
        Syp = Sxp if same_graph else shuffle_edges(Sy, rng, order, tolerance)
        ge += weight(Sxp[ia], Syp[ib]) >= obs
    return (1 + ge) / (1 + n_perm)


# ── who stands behind a country ──────────────────────────────────────────────

def reach(pairs: pd.DataFrame, country, depth: int = 2, direction: str = "upstream") -> pd.DataFrame:
    """The countries behind (upstream) or ahead of (downstream) `country` within `depth` steps.

    `pairs` has the columns actor, recipient, weight (support reported from actor to recipient).
    Upstream step 1 = who supports `country`; step 2 = who supports those. Downstream is the mirror image.
    A country is not visited twice, so loops do not repeat. Returns the used edges with a `step` column.
    """
    if direction not in ("upstream", "downstream"):
        raise ValueError("direction must be 'upstream' or 'downstream'")
    frontier, seen, parts = {country}, {country}, []
    for step in range(1, depth + 1):
        if direction == "upstream":
            rows = pairs[pairs.recipient.isin(frontier) & ~pairs.actor.isin(seen)]
            nxt = set(rows.actor)
        else:
            rows = pairs[pairs.actor.isin(frontier) & ~pairs.recipient.isin(seen)]
            nxt = set(rows.recipient)
        if rows.empty:
            break
        parts.append(rows.assign(step=step))
        frontier, seen = nxt, seen | nxt
    return pd.concat(parts, ignore_index=True) if parts else pairs.iloc[0:0].assign(step=[])


def pair_weights(events: pd.DataFrame, edges: pd.DataFrame, event_types: list, since: str | None = None,
                 until: str | None = None) -> pd.DataFrame:
    """Number of events per ordered pair (actor, recipient) for the given types, between `since` and `until`
    (inclusive, 'YYYY-MM-DD'). Columns: actor, recipient, weight."""
    ev = events[events.event_type.isin(event_types)]
    dates = pd.to_datetime(ev.event_date)
    if since:
        ev = ev[dates >= pd.Timestamp(since)]
        dates = pd.to_datetime(ev.event_date)
    if until:
        ev = ev[dates <= pd.Timestamp(until)]
    m = edges.merge(ev[["event_id"]], on="event_id")
    out = m.groupby(["actor_id", "recipient_id"]).size().rename("weight").reset_index()
    return out.rename(columns={"actor_id": "actor", "recipient_id": "recipient"})
