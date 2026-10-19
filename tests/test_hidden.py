"""Tests for src/hidden.py on synthetic worlds where the truth is known."""

import numpy as np
import pandas as pd
import pytest

import hidden as hd

N, T, SPLIT = 40, 30, 20


def world(boost=0.0, seed=0):
    """Busy countries produce more of everything (a volume confound). With boost > 0, a pair that had verbal
    cooperation before the split also gets extra material cooperation after it (a planted effect)."""
    rng = np.random.default_rng(seed)
    activity = rng.lognormal(0, 1.0, N)
    base = np.outer(activity, activity) * 0.012
    np.fill_diagonal(base, 0)
    words = rng.poisson(base[..., None] * np.ones(T) * 2.0)
    other = rng.poisson(base[..., None] * np.ones(T) * 3.0)
    lam_m = base[..., None] * np.ones(T) * 0.35
    material = rng.poisson(lam_m)
    if boost:
        had_words = words[..., :SPLIT].sum(axis=2) >= 3
        extra = rng.poisson(boost * had_words[..., None] * np.ones(T))
        material[..., SPLIT:] += extra[..., SPLIT:]
    everything = words + material + other
    return words, material, everything


def test_with_no_real_effect_the_volume_confound_is_removed_by_matching():
    w, m, e = world(boost=0.0)
    res = hd.prediction_test(w, m, e, SPLIT, k_verbal=3)
    assert res["ok"] and res["n_treated"] > 30
    assert res["rate_treated"] > res["rate_all_untreated"] + 0.1        # naive comparison: looks like an effect
    assert res["ci_low"] <= 0.0 <= res["ci_high"] or abs(res["difference"]) < 0.08   # matched: it is gone


def test_a_planted_effect_is_found_after_matching():
    w, m, e = world(boost=0.15)
    res = hd.prediction_test(w, m, e, SPLIT, k_verbal=3)
    assert res["difference"] > 0.1
    assert res["ci_low"] > 0.0


def test_matching_improves_the_covariate_balance():
    w, m, e = world()
    bal = hd.prediction_test(w, m, e, SPLIT)["balance"]
    assert (bal.smd_after.abs() < bal.smd_before.abs()).all()
    assert (bal.smd_after.abs() < 0.2).all()


def test_each_control_is_used_at_most_once():
    w, m, e = world()
    t = hd._pair_table(w, m, e, SPLIT)
    treated, pool = t[t.verbal_before >= 3].reset_index(drop=True), t[t.verbal_before < 3].reset_index(drop=True)
    idx = hd.match(treated, pool)
    assert len(set(idx)) == len(idx) == len(treated)


def test_matched_controls_have_fewer_than_k_verbal_events():
    w, m, e = world()
    t = hd._pair_table(w, m, e, SPLIT)
    treated, pool = t[t.verbal_before >= 3].reset_index(drop=True), t[t.verbal_before < 3].reset_index(drop=True)
    assert (pool.iloc[hd.match(treated, pool)].verbal_before < 3).all()


def test_only_new_keeps_pairs_without_earlier_material_support():
    w, m, e = world()
    all_pairs = hd.prediction_test(w, m, e, SPLIT)
    new_only = hd.prediction_test(w, m, e, SPLIT, only_new=True)
    assert new_only["n_treated"] < all_pairs["n_treated"] and new_only["only_new"]


def test_a_stricter_verbal_threshold_gives_fewer_treated_pairs():
    w, m, e = world()
    assert hd.prediction_test(w, m, e, SPLIT, k_verbal=6)["n_treated"] < hd.prediction_test(w, m, e, SPLIT, k_verbal=2)["n_treated"]


def test_too_few_treated_pairs_returns_not_ok_instead_of_a_number():
    w, m, e = world()
    assert hd.prediction_test(w, m, e, SPLIT, k_verbal=10_000)["ok"] is False


def test_the_test_is_reproducible_for_a_given_seed():
    w, m, e = world()
    a, b = hd.prediction_test(w, m, e, SPLIT, seed=3), hd.prediction_test(w, m, e, SPLIT, seed=3)
    assert a["difference"] == b["difference"] and a["ci_low"] == b["ci_low"]


# ── classification ───────────────────────────────────────────────────────────

def tiny():
    words, material = np.zeros((4, 4, 5)), np.zeros((4, 4, 5))
    words[0, 1, :] = 1            # 5 verbal events 0 -> 1
    material[0, 1, 2] = 2         # and material support: both
    words[0, 2, :3] = 1           # 3 verbal events 0 -> 2, no material: words only
    material[3, 0, 1] = 1         # material only 3 -> 0
    return words, material


def test_pairs_fall_into_the_four_classes():
    w, m = tiny()
    c = hd.classify_pairs(w, m, k_verbal=3, k_material=1).set_index(["a", "r"])
    assert c.loc[(0, 1), "class"] == hd.CLASS_BOTH
    assert c.loc[(0, 2), "class"] == hd.CLASS_WORDS
    assert c.loc[(3, 0), "class"] == hd.CLASS_MATERIAL
    assert c.loc[(1, 2), "class"] == hd.CLASS_NONE


def test_the_verbal_threshold_decides_between_words_only_and_neither():
    w, m = tiny()
    c = hd.classify_pairs(w, m, k_verbal=4).set_index(["a", "r"])
    assert c.loc[(0, 2), "class"] == hd.CLASS_NONE


def test_only_chains_whose_end_points_have_words_but_no_material_are_candidate_latent_ties():
    w, m = tiny()
    classes = hd.classify_pairs(w, m)
    nodes = ["Z", "Y", "X", "W"]                      # array positions 0..3
    chains = pd.DataFrame([{"a": "Z", "b": "Y", "c": "X", "weight": 5},      # Z -> X is words only
                           {"a": "Z", "b": "X", "c": "Y", "weight": 4},      # Z -> Y has material too
                           {"a": "W", "b": "Z", "c": "Y", "weight": 3}])     # W -> Y: neither
    out = hd.candidate_latent_ties(chains, classes, nodes)
    assert list(out.c) == ["X"] and out.iloc[0].ac_class == hd.CLASS_WORDS


# ── bootstrap and pools ──────────────────────────────────────────────────────

def test_the_cluster_bootstrap_is_wide_when_outcomes_differ_between_actors():
    diff = np.array([1.0] * 20 + [-1.0] * 20)                 # two actors, each with 20 pairs
    clusters = np.array([0] * 20 + [1] * 20)
    boot = hd.cluster_bootstrap(diff, clusters, n_boot=2000)
    assert np.percentile(boot, 2.5) <= -0.9 and np.percentile(boot, 97.5) >= 0.9
    # resampling single pairs would give a much narrower interval around 0
    pairwise = hd.cluster_bootstrap(diff, np.arange(40), n_boot=2000)
    assert np.percentile(pairwise, 97.5) - np.percentile(pairwise, 2.5) < 0.8


def test_the_cluster_bootstrap_of_a_constant_is_that_constant():
    boot = hd.cluster_bootstrap(np.full(30, 0.25), np.repeat(np.arange(6), 5))
    assert np.allclose(boot, 0.25)


def test_the_control_pool_holds_exactly_the_pairs_below_the_verbal_threshold():
    w, m, e = world()
    t = hd._pair_table(w, m, e, SPLIT)
    res = hd.prediction_test(w, m, e, SPLIT, k_verbal=3)
    assert res["n_pool"] == int((t.verbal_before < 3).sum())
    assert res["n_treated"] == int((t.verbal_before >= 3).sum())


# ── words clusters versus action clusters ────────────────────────────────────

def test_identical_partitions_agree_perfectly_and_beat_the_shuffle():
    m = np.repeat([0, 1, 2], 10)
    r = hd.agreement(m, np.repeat([5, 6, 7], 10), n_perm=300)
    assert r["ari"] == pytest.approx(1.0) and r["nmi"] == pytest.approx(1.0) and r["ari_p"] < 0.01


def test_unrelated_partitions_do_not_agree_more_than_chance():
    rng = np.random.default_rng(0)
    r = hd.agreement(rng.integers(0, 3, 60), rng.integers(0, 3, 60), n_perm=300)
    assert abs(r["ari"]) < 0.15 and r["ari_p"] > 0.05


def test_agreement_leaves_out_countries_without_a_cluster():
    r = hd.agreement(np.array([0, 0, 1, 1, -1]), np.array([0, 0, 1, 1, 2]), n_perm=50)
    assert r["n_countries"] == 4 and r["ari"] == pytest.approx(1.0)


def _two_layers(same_structure, seed=0):
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1, 2], 8)
    alt = np.repeat([0, 1, 2], 8)[rng.permutation(24)] if not same_structure else label
    mk = lambda lab: np.array([[0 if a == b else rng.poisson(3 if lab[a] == lab[b] else 0.1) for b in range(24)] for a in range(24)], float)   # noqa: E731
    return mk(label), mk(alt)


def test_bootstrap_agreement_is_high_when_both_layers_have_the_same_blocks_and_low_when_not():
    w, a = _two_layers(True)
    high = hd.agreement_bootstrap(w, a, n_boot=30)
    w2, a2 = _two_layers(False)
    low = hd.agreement_bootstrap(w2, a2, n_boot=30)
    assert np.mean(high) > 0.8 and np.mean(low) < 0.4


def test_deviation_is_zero_when_words_and_actions_have_the_same_partners():
    C = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], float)
    d = hd.deviation_scores(C, C.copy())
    assert np.allclose(d.deviation.dropna(), 0)


def test_deviation_is_one_when_the_partners_are_entirely_different():
    Cw = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], float)
    Ca = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], float)
    d = hd.deviation_scores(Cw, Ca)
    assert np.allclose(d.words_not_actions, 1) and np.allclose(d.actions_not_words, 1)


def test_deviation_is_split_into_the_two_directions():
    Cw = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], float)           # words: everyone with everyone
    Ca = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], float)           # actions: only 0 with 1
    d = hd.deviation_scores(Cw, Ca)
    assert d.words_not_actions[0] == pytest.approx(0.5)               # of its two words partners, one is not an action partner
    assert d.actions_not_words[0] == pytest.approx(0.0)               # its action partner is also a words partner


def test_pair_classes_cover_the_four_cases_and_the_uncertain_one():
    mw = np.array([0, 0, 1, 1, 2])
    ma = np.array([0, 0, 0, 1, 3])
    stab = np.array([1.0, 1.0, 1.0, 1.0, 0.2])                         # country 4 is unstable
    c = hd.pair_classes(mw, ma, stab, stab).set_index(["i", "j"])["class"]
    assert c[(0, 1)] == hd.CLASS_SAME_BOTH
    assert c[(2, 3)] == hd.CLASS_SAME_WORDS                              # same words cluster, different action clusters
    assert c[(0, 2)] == hd.CLASS_SAME_ACTIONS                            # same action cluster, different words clusters
    assert c[(1, 3)] == hd.CLASS_SAME_NEITHER
    assert c[(0, 4)] == hd.CLASS_UNCERTAIN


def test_a_lone_country_without_stability_makes_its_pairs_uncertain():
    m = np.array([0, 0, 1])
    stab = np.array([1.0, 1.0, np.nan])
    c = hd.pair_classes(m, m, stab, stab)
    assert (c[(c.i == 0) & (c.j == 2)]["class"] == hd.CLASS_UNCERTAIN).all()


def _cluster_world(effect, seed=0):
    rng = np.random.default_rng(seed)
    n = 40
    mem = np.repeat(np.arange(4), 10)
    act = rng.lognormal(0, 0.8, n)
    base = np.outer(act, act) * 0.05
    np.fill_diagonal(base, 0)
    everything_p1 = rng.poisson(base * 6).astype(float)
    words_p1 = rng.poisson(base * 2).astype(float)
    material_p1 = rng.poisson(base * 0.3).astype(float)
    boost = effect * (mem[:, None] == mem[None, :])
    material_p2 = rng.poisson(base * 0.3 + boost).astype(float)
    return mem, everything_p1, material_p1, material_p2, words_p1


def test_a_planted_cluster_effect_is_found_in_the_held_out_period():
    r = hd.cluster_prediction_test(*_cluster_world(0.12), only_new=False)
    assert r["ok"] and r["difference"] > 0.05 and r["ci_low"] > 0


def test_without_a_cluster_effect_the_matched_difference_is_about_zero():
    r = hd.cluster_prediction_test(*_cluster_world(0.0), only_new=False)
    assert r["ok"] and abs(r["difference"]) < 0.1 and r["ci_low"] <= 0.02


def test_cluster_prediction_new_links_only_uses_fewer_pairs():
    args = _cluster_world(0.05)
    a = hd.cluster_prediction_test(*args, only_new=False)
    b = hd.cluster_prediction_test(*args, only_new=True)
    assert b["n_treated"] < a["n_treated"] and b["only_new"]


def test_cluster_prediction_needs_enough_treated_pairs():
    mem, ev, m1, m2, w1 = _cluster_world(0.0)
    assert hd.cluster_prediction_test(np.arange(40), ev, m1, m2, w1)["ok"] is False      # nobody shares a cluster


def test_the_bootstrap_agreement_varies_because_the_events_are_re_drawn():
    rng = np.random.default_rng(3)
    lab = np.repeat([0, 1], 10)
    sparse = np.array([[0 if a == b else rng.poisson(0.9 if lab[a] == lab[b] else 0.4) for b in range(20)] for a in range(20)], float)
    boot = hd.agreement_bootstrap(sparse, sparse, n_boot=40, seed=1)
    assert np.nanstd(boot) > 0.02


def test_a_perfect_match_between_tiny_partitions_is_not_significant():
    """With four countries many shuffles reproduce the same partition, so equal values must count against us."""
    m = np.array([0, 0, 1, 1])
    r = hd.agreement(m, m.copy(), n_perm=400)
    assert r["ari"] == pytest.approx(1.0) and r["ari_p"] > 0.2


def _noisy_and_clean():
    rng = np.random.default_rng(3)
    lab = np.repeat([0, 1], 10)
    sparse = np.array([[0 if a == b else rng.poisson(0.9 if lab[a] == lab[b] else 0.4) for b in range(20)] for a in range(20)], float)
    dense = np.array([[0 if a == b else (50 if lab[a] == lab[b] else 0) for b in range(20)] for a in range(20)], float)
    return sparse, dense


def test_the_bootstrap_re_draws_the_words_layer():
    sparse, dense = _noisy_and_clean()
    assert np.nanstd(hd.agreement_bootstrap(sparse, dense, n_boot=40, seed=1)) > 0.02


def test_the_bootstrap_re_draws_the_actions_layer():
    sparse, dense = _noisy_and_clean()
    assert np.nanstd(hd.agreement_bootstrap(dense, sparse, n_boot=40, seed=1)) > 0.02
