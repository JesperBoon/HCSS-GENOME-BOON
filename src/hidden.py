"""Do words come before material support? A held-out test, and candidate "latent" ties.

Question
  If country A has a verbal cooperation tie with country R in the first part of the period (statements,
  talks, agreements, pledges: the words layer), does it later get a MATERIAL cooperation tie with R (aid,
  joint projects) more often than comparable pairs that had no such verbal tie?

Why matching
  Busy countries have more of everything: more words, and later more aid. Comparing a pair with a verbal tie
  with a random pair would only rediscover that. So every treated pair is matched to the untreated pair
  that is most similar in volume (the actor's other events, the recipient's other events, and the pair's own
  other events) before the outcomes are compared. The result is an association after matching on volume,
  not proof that words cause aid.

Words alone prove nothing about a hidden path: a verbal tie without an observed material tie is "empty
words" or a "candidate latent tie", not a secret. Latent means: not observed in this English-language
newswire corpus.
"""

import numpy as np
import pandas as pd

CLASS_BOTH = "Words and material support"
CLASS_WORDS = "Words only (candidate latent tie or empty words)"
CLASS_MATERIAL = "Material only (silent support)"
CLASS_NONE = "Neither"


# ── features and matching ────────────────────────────────────────────────────

def _pair_table(words: np.ndarray, material: np.ndarray, everything: np.ndarray, split: int) -> pd.DataFrame:
    """One row per ordered pair a -> r with counts before and after `split`, and the volume covariates
    measured before the split. `words` and `material` are cooperation tensors, `everything` holds all events."""
    n = words.shape[0]
    a, r = np.nonzero(~np.eye(n, dtype=bool))
    before, after = slice(0, split), slice(split, None)
    ev_before = everything[..., before].sum(axis=2)
    pair_events = ev_before[a, r]
    verbal = words[..., before].sum(axis=2)[a, r]
    out_total = ev_before.sum(axis=1)[a]            # everything the actor did, in the first period
    in_total = ev_before.sum(axis=0)[r]             # everything done to the recipient
    return pd.DataFrame({
        "a": a, "r": r,
        "verbal_before": verbal, "material_before": material[..., before].sum(axis=2)[a, r],
        "material_after": material[..., after].sum(axis=2)[a, r],
        "verbal_after": words[..., after].sum(axis=2)[a, r],
        "cov_actor": np.log1p(out_total - pair_events),
        "cov_recipient": np.log1p(in_total - pair_events),
        "cov_pair": np.log1p(pair_events - verbal),
    })


COVARIATES = ["cov_actor", "cov_recipient", "cov_pair"]


def match(treated: pd.DataFrame, pool: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Greedy 1:1 nearest-neighbour matching without replacement on the standardised covariates.
    Returns, for each treated row, the index (in `pool`'s order) of its control."""
    rng = np.random.default_rng(seed)
    Z = pd.concat([treated[COVARIATES], pool[COVARIATES]])
    sd = Z.std().replace(0, 1)
    zt = (treated[COVARIATES] / sd).to_numpy()
    zc = (pool[COVARIATES] / sd).to_numpy()
    free = np.ones(len(zc), dtype=bool)
    chosen = np.empty(len(zt), dtype=int)
    for i in rng.permutation(len(zt)):
        d = ((zc - zt[i]) ** 2).sum(axis=1)
        d[~free] = np.inf
        j = int(d.argmin())
        chosen[i] = j
        free[j] = False
    return chosen


def cluster_bootstrap(diff: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean of `diff`, resampling whole clusters (here: actor countries) with
    replacement. Pairs that share an actor are not independent, so resampling single pairs would make the
    confidence interval too narrow."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    sizes = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
    return (draws @ sums) / (draws @ sizes)


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference of a covariate between two groups."""
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


# ── the held-out prediction test ─────────────────────────────────────────────

def prediction_test(words: np.ndarray, material: np.ndarray, everything: np.ndarray, split: int,
                    k_verbal: int = 3, outcome_min: int = 1, only_new: bool = False,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Treated = pairs with at least `k_verbal` verbal cooperation events before `split`.
    Outcome = at least `outcome_min` material cooperation events after `split`.
    only_new=True keeps only pairs without any material cooperation before the split, so the question
    becomes: does a verbal tie predict a NEW material tie?

    Returns the outcome rates, the difference after matching with a confidence interval (bootstrap over
    actor countries, because pairs of one actor are not independent), the covariate balance and the counts.
    """
    t = _pair_table(words, material, everything, split)
    if only_new:
        t = t[t.material_before == 0]
    t["y"] = (t.material_after >= outcome_min).astype(int)
    treated = t[t.verbal_before >= k_verbal].reset_index(drop=True)
    pool = t[t.verbal_before < k_verbal].reset_index(drop=True)
    res = {"n_treated": len(treated), "n_pool": len(pool), "k_verbal": k_verbal, "only_new": only_new}
    if len(treated) < 5 or len(pool) < len(treated):
        return {**res, "ok": False}
    idx = match(treated, pool, seed)
    matched = pool.iloc[idx].reset_index(drop=True)

    diff = treated.y.to_numpy() - matched.y.to_numpy()
    boot = cluster_bootstrap(diff, treated.a.to_numpy(), n_boot, seed + 1)
    res.update({
        "ok": True, "rate_treated": float(treated.y.mean()), "rate_matched": float(matched.y.mean()),
        "rate_all_untreated": float(pool.y.mean()),
        "difference": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "balance": pd.DataFrame({
            "covariate": COVARIATES,
            "smd_before": [_smd(treated[c], pool[c]) for c in COVARIATES],
            "smd_after": [_smd(treated[c], matched[c]) for c in COVARIATES]}),
    })
    return res


# ── 2x2 classification and candidate latent ties ─────────────────────────────

def classify_pairs(words: np.ndarray, material: np.ndarray, k_verbal: int = 3, k_material: int = 1) -> pd.DataFrame:
    """Every ordered pair that has at least one cooperation event, over the whole period, in one of four classes."""
    n = words.shape[0]
    a, r = np.nonzero(~np.eye(n, dtype=bool))
    v, m = words.sum(axis=2)[a, r], material.sum(axis=2)[a, r]
    has_v, has_m = v >= k_verbal, m >= k_material
    cls = np.select([has_v & has_m, has_v & ~has_m, ~has_v & has_m], [CLASS_BOTH, CLASS_WORDS, CLASS_MATERIAL],
                    default=CLASS_NONE)
    return pd.DataFrame({"a": a, "r": r, "verbal": v, "material": m, "class": cls})


def candidate_latent_ties(chain_table: pd.DataFrame, classes: pd.DataFrame, nodes: list) -> pd.DataFrame:
    """Chains A -> B -> C whose end points A and C have a verbal tie but no observed material tie.

    The chain says support passes through B; the words say A and C are close; nothing material is
    observed between A and C directly. That makes A -> C a candidate latent tie: a reason to look closer,
    not a finding. `chain_table` needs the columns a, b, c holding country ids.
    """
    idx = {c: i for i, c in enumerate(nodes)}
    lookup = classes.set_index(["a", "r"])
    rows = []
    for ch_ in chain_table.itertuples():
        key = (idx[ch_.a], idx[ch_.c])
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        rows.append({**ch_._asdict(), "ac_verbal": row.verbal, "ac_material": row.material, "ac_class": row["class"]})
    out = pd.DataFrame(rows)
    return out[out.ac_class == CLASS_WORDS] if len(out) else out


# ── words clusters versus action clusters ────────────────────────────────────

CLASS_SAME_BOTH = "Same words cluster and same action cluster"
CLASS_SAME_WORDS = "Same words cluster only (candidate latent tie or empty words)"
CLASS_SAME_ACTIONS = "Same action cluster only (silent support)"
CLASS_SAME_NEITHER = "Neither"
CLASS_UNCERTAIN = "Uncertain (a country is not stable in one of the two)"


def _cl():
    import clusters   # imported here: clusters needs igraph, hidden's older functions do not
    return clusters


def agreement(mem_words: np.ndarray, mem_actions: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """How alike are the words clusters and the action clusters over the same countries?

    Adjusted Rand index and normalised mutual information, each with an empirical p-value from shuffling the
    action labels over the countries (what would two unrelated partitions of this shape give?).
    """
    c = _cl()
    keep = (mem_words >= 0) & (mem_actions >= 0)
    a, b = mem_words[keep], mem_actions[keep]
    rng = np.random.default_rng(seed)
    ari, nmi = c.adjusted_rand(a, b), c.normalized_mutual_information(a, b)
    null_ari = np.array([c.adjusted_rand(a, rng.permutation(b)) for _ in range(n_perm)])
    null_nmi = np.array([c.normalized_mutual_information(a, rng.permutation(b)) for _ in range(n_perm)])
    return {"n_countries": int(keep.sum()), "ari": ari, "nmi": nmi,
            "ari_null_mean": float(null_ari.mean()), "nmi_null_mean": float(null_nmi.mean()),
            "ari_p": float((1 + (null_ari >= ari).sum()) / (1 + n_perm)),
            "nmi_p": float((1 + (null_nmi >= nmi).sum()) / (1 + n_perm))}


def agreement_bootstrap(counts_words: np.ndarray, counts_actions: np.ndarray, n_boot: int = 200, seed: int = 0,
                        resolution: float = 1.0, log: bool = False) -> np.ndarray:
    """Adjusted Rand indices over re-drawn event counts (Poisson), each time clustering both layers again.
    The spread is the uncertainty that comes from the reported events being a sample."""
    c = _cl()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_boot):
        w = c.leiden_once(c.transform(rng.poisson(counts_words), log), resolution, seed + i)
        a = c.leiden_once(c.transform(rng.poisson(counts_actions), log), resolution, seed + i)
        out.append(c.adjusted_rand(w, a))
    return np.array(out)


def deviation_scores(coassign_words: np.ndarray, coassign_actions: np.ndarray) -> pd.DataFrame:
    """Per country: how often its words partners are not its action partners, and the reverse.

    words_not_actions = share of the (co-assignment weighted) words partners that are not action partners;
    actions_not_words likewise. Uses the co-assignment matrices of the stability runs, so a partner counts as much
    as the clusters agree on it. Countries without partners in a layer get NaN.
    """
    Cw, Ca = np.nan_to_num(coassign_words.copy()), np.nan_to_num(coassign_actions.copy())
    np.fill_diagonal(Cw, 0)
    np.fill_diagonal(Ca, 0)
    sw, sa = Cw.sum(axis=1), Ca.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w_not_a = (Cw * (1 - Ca)).sum(axis=1) / sw
        a_not_w = (Ca * (1 - Cw)).sum(axis=1) / sa
    w_not_a[sw == 0] = np.nan
    a_not_w[sa == 0] = np.nan
    return pd.DataFrame({"words_not_actions": w_not_a, "actions_not_words": a_not_w,
                         "deviation": np.nanmean(np.vstack([w_not_a, a_not_w]), axis=0)})


def pair_classes(mem_words: np.ndarray, mem_actions: np.ndarray, stab_words: np.ndarray, stab_actions: np.ndarray,
                 min_stability: float = 0.7) -> pd.DataFrame:
    """Every unordered pair of countries in one of the classes above, from cluster co-membership.

    A pair is 'uncertain' if either country is clustered with a stability below the threshold in either layer.
    """
    n = len(mem_words)
    i, j = np.triu_indices(n, k=1)
    clustered = (mem_words[i] >= 0) & (mem_words[j] >= 0) & (mem_actions[i] >= 0) & (mem_actions[j] >= 0)
    s = lambda arr: np.nan_to_num(arr, nan=0.0)   # noqa: E731  (a lone country has no stability: treat as uncertain)
    stable = (np.minimum.reduce([s(stab_words)[i], s(stab_words)[j], s(stab_actions)[i], s(stab_actions)[j]]) >= min_stability)
    same_w = mem_words[i] == mem_words[j]
    same_a = mem_actions[i] == mem_actions[j]
    cls = np.select([~clustered | ~stable, same_w & same_a, same_w & ~same_a, ~same_w & same_a],
                    [CLASS_UNCERTAIN, CLASS_SAME_BOTH, CLASS_SAME_WORDS, CLASS_SAME_ACTIONS], default=CLASS_SAME_NEITHER)
    return pd.DataFrame({"i": i, "j": j, "class": cls})


def cluster_prediction_test(mem_words_p1: np.ndarray, everything_p1: np.ndarray, material_p1: np.ndarray,
                            material_p2: np.ndarray, words_p1: np.ndarray, only_new: bool = False,
                            outcome_min: int = 1, n_boot: int = 2000, seed: int = 0) -> dict:
    """Held-out test. Words clusters are built on period 1; do two countries in the same words cluster get a material
    cooperation tie (either direction) in period 2 more often than the most similar pairs that did not share a cluster?

    `everything_p1`, `material_p1`, `words_p1`, `material_p2` are (n, n) event counts. Pairs are unordered and
    matched on the activity of the busier and of the quieter country and of the pair's other events in period 1;
    the confidence interval resamples whole countries (by the smaller country id of each pair).
    """
    n = len(mem_words_p1)
    i, j = np.triu_indices(n, k=1)
    ok = (mem_words_p1[i] >= 0) & (mem_words_p1[j] >= 0)
    i, j = i[ok], j[ok]
    tot = everything_p1.sum(axis=0) + everything_p1.sum(axis=1)
    pair_all = everything_p1[i, j] + everything_p1[j, i]
    pair_words = words_p1[i, j] + words_p1[j, i]
    before = material_p1[i, j] + material_p1[j, i]
    after = material_p2[i, j] + material_p2[j, i]
    t = pd.DataFrame({"a": i, "r": j, "treated": mem_words_p1[i] == mem_words_p1[j],
                      "cov_actor": np.log1p(np.maximum(tot[i], tot[j])), "cov_recipient": np.log1p(np.minimum(tot[i], tot[j])),
                      "cov_pair": np.log1p(pair_all - pair_words), "material_before": before,
                      "y": (after >= outcome_min).astype(int)})
    if only_new:
        t = t[t.material_before == 0]
    treated, pool = t[t.treated].reset_index(drop=True), t[~t.treated].reset_index(drop=True)
    res = {"n_treated": len(treated), "n_pool": len(pool), "only_new": only_new, "outcome_min": outcome_min}
    if len(treated) < 5 or len(pool) < len(treated):
        return {**res, "ok": False}
    idx = match(treated, pool, seed)
    matched = pool.iloc[idx].reset_index(drop=True)
    diff = treated.y.to_numpy() - matched.y.to_numpy()
    boot = cluster_bootstrap(diff, treated.a.to_numpy(), n_boot, seed + 1)
    res.update({"ok": True, "rate_treated": float(treated.y.mean()), "rate_matched": float(matched.y.mean()),
                "rate_all_untreated": float(pool.y.mean()), "difference": float(diff.mean()),
                "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                "max_abs_smd_after": float(max(abs(_smd(treated[c], matched[c])) for c in COVARIATES))})
    return res
