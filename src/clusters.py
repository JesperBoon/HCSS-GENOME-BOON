"""Support clusters: groups of countries that support each other, and the brokers between them.

What this is, and what it is not
  A cluster here is a statistical pattern in REPORTED events: a group of countries with many more reported
  support events (or words, or conflict events) among themselves than the number of reports each country
  receives would lead us to expect. It is not proof of a formal alliance, not a flow of money or weapons (GENOME has
  no amounts), and closeness in time is not causation. "Latent" means "not observed in this English-language news
  corpus", not secret.

Method in short
  1. A directed, weighted graph per layer (words / actions), category (cooperation / conflict) and period; weight =
     number of events (optionally log(1 + n), so a few big countries do not dominate).
  2. Leiden community detection on the directed graph (Reichardt-Bornholdt configuration model, so the number of ties of
     each country is accounted for), with a resolution parameter.
  3. Stability: 100 runs with different seeds give a co-assignment matrix (how often two countries share a cluster); the
     consensus partition is built from it, and each country gets a stability score.
  4. Brokers: the participation coefficient, how evenly a country spreads its ties over clusters.
  5. Null model: a directed configuration model that keeps every country's number of events given and received, 1,000
     times; modularity, largest cluster and broker scores are compared with it.
"""

import warnings
from dataclasses import dataclass

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd

import chains as ch


# ── building the matrix ──────────────────────────────────────────────────────

def weights_matrix(pairs: pd.DataFrame, nodes: list) -> np.ndarray:
    """Event counts per ordered pair as an (n, n) array. `pairs` has the columns actor, recipient, weight."""
    idx = {c: i for i, c in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for r in pairs.itertuples():
        if r.actor in idx and r.recipient in idx and r.actor != r.recipient:
            W[idx[r.actor], idx[r.recipient]] += r.weight
    return W


def transform(W: np.ndarray, log: bool) -> np.ndarray:
    """log(1 + n) damps the biggest pairs; without it raw counts are used."""
    return np.log1p(W) if log else W


# ── one Leiden run ───────────────────────────────────────────────────────────

def directed_modularity(W: np.ndarray, membership: np.ndarray) -> float:
    """Leicht-Newman modularity of a directed weighted graph; countries with membership -1 are left out."""
    keep = membership >= 0
    W, m = W[np.ix_(keep, keep)], membership[keep]
    total = W.sum()
    if total == 0:
        return 0.0
    same = m[:, None] == m[None, :]
    expected = np.outer(W.sum(axis=1), W.sum(axis=0)) / total
    return float(((W - expected) * same).sum() / total)


def leiden_once(W: np.ndarray, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    """Cluster label per country (-1 for countries without any tie). Runs Leiden until nothing improves.

    Note: use `partition.quality()` as the objective, not `.modularity` (which ignores direction).
    """
    n = W.shape[0]
    active = np.flatnonzero((W.sum(axis=0) + W.sum(axis=1)) > 0)
    membership = -np.ones(n, dtype=int)
    if len(active) == 0:
        return membership
    sub = W[np.ix_(active, active)]
    src, dst = np.nonzero(sub)
    g = ig.Graph(n=len(active), edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    g.es["weight"] = sub[src, dst].tolist()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                             resolution_parameter=resolution, n_iterations=-1, seed=seed)
    membership[active] = part.membership
    return membership


def canonical(membership: np.ndarray) -> np.ndarray:
    """Relabel clusters 0, 1, 2... by size (biggest first); -1 stays -1."""
    out = -np.ones_like(membership)
    labels, counts = np.unique(membership[membership >= 0], return_counts=True)
    order = sorted(zip(labels, counts), key=lambda lc: (-lc[1], np.flatnonzero(membership == lc[0]).min()))
    for new, (old, _) in enumerate(order):
        out[membership == old] = new
    return out


# ── stability and consensus ──────────────────────────────────────────────────

def coassignment(runs: np.ndarray) -> np.ndarray:
    """(n, n) share of runs in which two countries are in the same cluster; NaN if they were never both active."""
    R, n = runs.shape
    both = np.zeros((n, n))
    same = np.zeros((n, n))
    for r in runs:
        act = r >= 0
        both += np.outer(act, act)
        same += (r[:, None] == r[None, :]) & np.outer(act, act)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(both > 0, same / both, np.nan)


@dataclass
class Consensus:
    membership: np.ndarray      # cluster per country, -1 = no ties in the period
    stability: np.ndarray       # mean co-assignment with the other members of its own cluster (NaN for a lone country)
    coassign: np.ndarray        # (n, n)
    run_quality: np.ndarray     # directed modularity (on the observed counts) of the partition of each run
    observed_quality: float     # median modularity of Leiden runs on the observed counts themselves (for the null model)
    modularity: float           # of the consensus partition
    resolution: float


def consensus(counts: np.ndarray, resolution: float = 1.0, n_runs: int = 100, seed: int = 0,
              threshold: float = 0.2, log: bool = False, resample: bool = True) -> Consensus:
    """Run Leiden `n_runs` times, build the co-assignment matrix and cluster it again (Lancichinetti-Fortunato).

    `counts` are event counts per ordered pair; `log` clusters on log(1 + n). Every run uses its own random seed and,
    with `resample=True`, its own Poisson re-draw of the event counts: reported events are a sample of what happened,
    and a country that is tied about equally to two blocks then flips between them. With `resample=False` only the
    seed varies, and Leiden settles such a tie the same way every time (a stability of 1.0 that says little).
    Entries of the co-assignment matrix below `threshold` are ignored, so a country that only sometimes shares a
    cluster with another does not glue them together.
    """
    n = counts.shape[0]
    W = transform(counts, log)
    rng = np.random.default_rng(seed)
    runs = np.array([leiden_once(transform(rng.poisson(counts) if resample else counts, log), resolution, seed + i)
                     for i in range(n_runs)])
    run_q = np.array([directed_modularity(transform(counts, log), r) for r in runs])
    active_any = (W.sum(axis=0) + W.sum(axis=1)) > 0
    runs[:, ~active_any] = -1
    C = coassignment(runs)
    active = np.flatnonzero(active_any)
    membership = -np.ones(n, dtype=int)
    if len(active):
        Cs = np.nan_to_num(C[np.ix_(active, active)])
        Cs = np.where(Cs >= threshold, Cs, 0.0)
        np.fill_diagonal(Cs, 0.0)
        src, dst = np.nonzero(np.triu(Cs))
        g = ig.Graph(n=len(active), edges=list(zip(src.tolist(), dst.tolist())))
        g.es["weight"] = Cs[src, dst].tolist()
        part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", n_iterations=-1, seed=seed)
        membership[active] = part.membership
    membership = canonical(membership)
    stability = np.full(n, np.nan)
    for i in np.flatnonzero(membership >= 0):
        mates = np.flatnonzero((membership == membership[i]) & (np.arange(n) != i))
        if len(mates):
            stability[i] = float(np.nanmean(C[i, mates]))
    obs_q = float(np.median([directed_modularity(W, leiden_once(W, resolution, seed + 10_000 + i)) for i in range(10)]))
    return Consensus(membership, stability, C, run_q, obs_q, directed_modularity(W, membership), resolution)


# ── brokers, flows ───────────────────────────────────────────────────────────

def participation(W: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Participation coefficient per country: 1 - sum over clusters of (share of its ties in that cluster)^2.
    0 = all ties inside one cluster; close to 1 = ties spread evenly over many clusters (a bridge)."""
    S = W + W.T
    out = np.full(W.shape[0], np.nan)
    labels = np.unique(membership[membership >= 0])
    for i in range(W.shape[0]):
        k = S[i, membership >= 0].sum() if membership[i] >= 0 else 0
        if k > 0:
            out[i] = 1 - sum((S[i, membership == c].sum() / k) ** 2 for c in labels)
    return out


def outside_share(W: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Share of each country's ties (given + received) that go to other clusters."""
    S = W + W.T
    out = np.full(W.shape[0], np.nan)
    for i in np.flatnonzero(membership >= 0):
        k = S[i, membership >= 0].sum()
        if k > 0:
            out[i] = 1 - S[i, membership == membership[i]].sum() / k
    return out


def flows(W: np.ndarray, membership: np.ndarray) -> pd.DataFrame:
    """Total events from cluster X (rows) to cluster Y (columns); the diagonal is inside a cluster."""
    labels = np.unique(membership[membership >= 0])
    F = np.array([[W[np.ix_(membership == x, membership == y)].sum() for y in labels] for x in labels])
    return pd.DataFrame(F, index=labels, columns=labels)


# ── the null model ───────────────────────────────────────────────────────────

def draw_null(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Directed configuration model: shuffle which recipient goes with which actor among all events, so every country
    keeps exactly the number of events it gives and receives. Self-loops are repaired by swapping."""
    n = counts.shape[0]
    a_idx, r_idx = np.nonzero(counts)
    mult = np.rint(counts[a_idx, r_idx]).astype(int)
    A, R = np.repeat(a_idx, mult), np.repeat(r_idx, mult)
    R = rng.permutation(R)
    for _ in range(60):
        bad = np.flatnonzero(A == R)
        if len(bad) == 0:
            break
        partners = rng.integers(0, len(R), size=len(bad))
        for i, j in zip(bad, partners):
            if A[i] != R[j] and A[j] != R[i]:
                R[i], R[j] = R[j], R[i]
    M = np.bincount(A * n + R, minlength=n * n).reshape(n, n).astype(float)
    np.fill_diagonal(M, 0)
    return M


def null_distribution(counts: np.ndarray, n_null: int = 1000, resolution: float = 1.0, log: bool = False,
                      seed: int = 0) -> dict:
    """Cluster `n_null` configuration-model graphs once each. Returns arrays over draws:
    modularity, largest cluster's share of the countries, number of clusters, and (n_null, n) participation."""
    rng = np.random.default_rng(seed)
    q, big, k, part = [], [], [], []
    for i in range(n_null):
        M = transform(draw_null(counts, rng), log)
        m = leiden_once(M, resolution, seed=int(rng.integers(1 << 30)))
        act = m >= 0
        q.append(directed_modularity(M, m))
        sizes = np.bincount(m[act]) if act.any() else np.array([0])
        big.append(sizes.max() / max(act.sum(), 1))
        k.append(len(np.unique(m[act])))
        part.append(participation(M, m))
    return {"modularity": np.array(q), "largest_share": np.array(big), "n_clusters": np.array(k),
            "participation": np.array(part)}


def compare_with_null(cons: Consensus, W: np.ndarray, null: dict) -> dict:
    """Observed against the null: z and empirical p for modularity (median of the runs, to compare like with like)
    and largest cluster, and a z-score per country for the participation coefficient."""
    obs_q = cons.observed_quality
    act = cons.membership >= 0
    sizes = np.bincount(cons.membership[act]) if act.any() else np.array([0])
    obs_big = sizes.max() / max(act.sum(), 1)
    nq = null["modularity"]
    p_q = (1 + (nq >= obs_q).sum()) / (1 + len(nq))
    obs_part = participation(W, cons.membership)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # countries without ties in some draws: empty slices
        sd = np.nanstd(null["participation"], axis=0)
        z_part = (obs_part - np.nanmean(null["participation"], axis=0)) / np.where(sd > 0, sd, np.nan)
    p_part = np.array([(1 + np.nansum(null["participation"][:, i] >= obs_part[i])) / (1 + np.isfinite(null["participation"][:, i]).sum())
                       if np.isfinite(obs_part[i]) else np.nan for i in range(W.shape[0])])
    return {"modularity": obs_q, "modularity_null_mean": float(nq.mean()), "modularity_z": float((obs_q - nq.mean()) / (nq.std() or np.nan)),
            "modularity_p": float(p_q), "largest_share": float(obs_big),
            "largest_share_null_mean": float(null["largest_share"].mean()),
            "participation": obs_part, "participation_z": z_part, "participation_p": p_part}


# ── hubs, opponents, sensitivity ─────────────────────────────────────────────

def hubs(W: np.ndarray, k: int = 1) -> np.ndarray:
    """Indices of the k countries with the most events given plus received."""
    return np.argsort(-(W.sum(axis=0) + W.sum(axis=1)))[:k]


def without(W: np.ndarray, drop) -> np.ndarray:
    """A copy of the matrix in which the given countries have no ties (they come out as -1, 'not clustered')."""
    M = W.copy()
    M[list(drop), :] = 0
    M[:, list(drop)] = 0
    return M


def conflict_between_clusters(membership: np.ndarray, conflict: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """Do adversaries sit in different blocks? Share of conflict events between countries of the SAME cluster,
    against the share expected if the cluster labels were handed out at random (sizes kept)."""
    act = np.flatnonzero(membership >= 0)
    K = conflict[np.ix_(act, act)].astype(float)
    total = K.sum()
    if total == 0:
        return {"within_share": np.nan, "expected": np.nan, "z": np.nan, "p_low": np.nan, "events": 0}
    labels = membership[act]
    within = (K * (labels[:, None] == labels[None, :])).sum() / total
    rng = np.random.default_rng(seed)
    null = np.array([(K * ((p := rng.permutation(labels))[:, None] == p[None, :])).sum() / total for _ in range(n_perm)])
    sd = null.std()
    return {"within_share": float(within), "expected": float(null.mean()),
            "z": float((within - null.mean()) / sd) if sd > 0 else np.nan,
            "p_low": float((1 + (null <= within).sum()) / (1 + n_perm)), "events": int(total)}


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two partitions of the same items (countries with -1 in either are left out)."""
    keep = (a >= 0) & (b >= 0)
    a, b = a[keep], b[keep]
    n = len(a)
    if n < 2:
        return np.nan
    table = pd.crosstab(a, b).to_numpy()
    comb = lambda x: x * (x - 1) / 2   # noqa: E731
    sum_ij = comb(table).sum()
    sum_a, sum_b = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb(n)
    maximum = (sum_a + sum_b) / 2
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information (arithmetic-mean normalisation), 0 = independent, 1 = identical."""
    keep = (a >= 0) & (b >= 0)
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 0:
        return np.nan
    table = pd.crosstab(a, b).to_numpy() / n
    pa, pb = table.sum(axis=1), table.sum(axis=0)
    nz = table > 0
    mi = (table[nz] * np.log(table[nz] / np.outer(pa, pb)[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / ((ha + hb) / 2)) if (ha + hb) > 0 else 1.0


def align_over_time(memberships: list) -> list:
    """Give clusters the same label in consecutive periods when they share most of their countries (greedy match by
    Jaccard overlap, at least 0.2); other clusters get a new label. Returns one label array per period."""
    aligned, next_label, prev = [], 0, None
    for m in memberships:
        out = -np.ones_like(m)
        cur = {c: set(np.flatnonzero(m == c)) for c in np.unique(m[m >= 0])}
        pairs = []
        if prev is not None:
            for c, s in cur.items():
                for pl, ps in prev.items():
                    jac = len(s & ps) / len(s | ps)
                    if jac >= 0.2:
                        pairs.append((jac, c, pl))
        used_c, used_p = set(), set()
        for jac, c, pl in sorted(pairs, reverse=True):
            if c not in used_c and pl not in used_p:
                out[list(cur[c])] = pl
                used_c.add(c)
                used_p.add(pl)
        for c, s in cur.items():
            if c not in used_c:
                out[list(s)] = next_label
                next_label += 1
        # never reuse a label for two clusters in one period, and keep the counter ahead of all labels in use
        next_label = max(next_label, int(out.max()) + 1 if (out >= 0).any() else next_label)
        aligned.append(out)
        prev = {c: set(np.flatnonzero(out == c)) for c in np.unique(out[out >= 0])}
    return aligned


# ── drill-down: chains inside or between clusters ────────────────────────────

def chains_between_clusters(counts_t: np.ndarray, nodes: list, membership: np.ndarray, x: int, y: int | None = None,
                            names: dict | None = None, top: int = 25) -> pd.DataFrame:
    """Two-step paths A -> B -> C in the window graph `counts_t[a, r, t]` (order known only at window level).

    With y=None all three countries are in cluster x; otherwise A is in x and C is in y (B anywhere). Weight is the
    weaker of the two links, summed over windows (`same` window: order unknown; `next` window: ordered).
    """
    same, nxt = ch.chain_weights(counts_t, counts_t)
    weight = same + nxt
    in_x, in_y = membership == x, membership == (x if y is None else y)
    mask = in_x[:, None, None] & in_y[None, None, :]
    if y is None:
        mask = mask & in_x[None, :, None]
    A, B, C = np.nonzero((weight > 0) & mask)
    names = names or {}
    df = pd.DataFrame({"a": [nodes[i] for i in A], "b": [nodes[i] for i in B], "c": [nodes[i] for i in C],
                       "weight": weight[A, B, C], "same_window": same[A, B, C], "next_window": nxt[A, B, C]})
    df["path"] = df.a.map(lambda c: names.get(c, c)) + " → " + df.b.map(lambda c: names.get(c, c)) + " → " + df.c.map(lambda c: names.get(c, c))
    return df.sort_values("weight", ascending=False).head(top).reset_index(drop=True)
