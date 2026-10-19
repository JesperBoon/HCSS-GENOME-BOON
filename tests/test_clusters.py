"""Tests for src/clusters.py on small graphs whose answer is known."""

import numpy as np
import pandas as pd
import pytest

import clusters as cl


def blocks(sizes=(8, 8, 8), p_in=3.0, p_out=0.15, seed=0):
    """Directed event counts: dense inside each block, sparse between blocks."""
    rng = np.random.default_rng(seed)
    label = np.repeat(np.arange(len(sizes)), sizes)
    n = len(label)
    W = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                W[a, b] = rng.poisson(p_in if label[a] == label[b] else p_out)
    return W, label


# ── building and scoring ─────────────────────────────────────────────────────

def test_weights_matrix_places_counts_and_ignores_unknown_countries_and_self_pairs():
    pairs = pd.DataFrame({"actor": [10, 10, 20, 99, 10], "recipient": [20, 20, 10, 10, 10], "weight": [2, 1, 4, 7, 5]})
    W = cl.weights_matrix(pairs, [10, 20])
    assert W.tolist() == [[0, 3], [4, 0]]


def test_log_transform_damps_big_counts():
    W = np.array([[0, 100.0], [1.0, 0]])
    assert cl.transform(W, True)[0, 1] / cl.transform(W, True)[1, 0] < 100 / 1
    assert np.array_equal(cl.transform(W, False), W)


def test_directed_modularity_matches_a_hand_worked_example():
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = W[2, 3] = W[3, 2] = 2
    # m = 8; every pair expects 2*2/8 = 0.5; inside-cluster weight 8, expected 8*0.5 = 4 -> (8-4)/8
    assert cl.directed_modularity(W, np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert cl.directed_modularity(W, np.array([0, 0, 0, 0])) == pytest.approx(0.0)


def test_modularity_ignores_countries_without_a_cluster():
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 2
    assert cl.directed_modularity(W, np.array([0, 0, -1, -1])) == pytest.approx(0.0)


# ── one run, canonical labels, consensus ─────────────────────────────────────

def test_leiden_recovers_planted_blocks():
    W, label = blocks()
    assert cl.adjusted_rand(cl.leiden_once(W, seed=1), label) == pytest.approx(1.0)


def test_countries_without_any_tie_are_not_clustered():
    W, _ = blocks(sizes=(6, 6))
    W = np.pad(W, ((0, 2), (0, 2)))                    # two isolated countries
    m = cl.leiden_once(W)
    assert (m[-2:] == -1).all() and (m[:-2] >= 0).all()


def test_an_empty_graph_gives_no_clusters():
    assert (cl.leiden_once(np.zeros((5, 5))) == -1).all()
    assert (cl.consensus(np.zeros((5, 5)), n_runs=3).membership == -1).all()


def test_canonical_labels_run_from_the_biggest_cluster():
    assert cl.canonical(np.array([5, 5, 2, 9, 9, 9, -1])).tolist() == [1, 1, 2, 0, 0, 0, -1]


def test_clear_blocks_are_found_with_full_stability():
    W, label = blocks()
    c = cl.consensus(W, n_runs=40)
    assert cl.adjusted_rand(c.membership, label) == pytest.approx(1.0)
    assert np.nanmin(c.stability) > 0.95
    assert c.modularity > 0.4


def test_a_country_tied_equally_to_two_blocks_is_unstable():
    W, label = blocks(sizes=(8, 8, 8))
    n = W.shape[0]
    W = np.pad(W, ((0, 1), (0, 1)))
    W[n, :8] = 3; W[:8, n] = 3                          # the new country is tied equally to block 0 ...
    W[n, 8:16] = 3; W[8:16, n] = 3                      # ... and to block 1
    c = cl.consensus(W, n_runs=80, seed=3)
    others = np.delete(c.stability, n)
    assert c.stability[n] < 0.8 < np.nanmin(others)


def test_without_resampling_a_tie_is_settled_the_same_way_every_time():
    """Only the seed varies: Leiden breaks the tie identically in every run, so the stability says nothing."""
    W, _ = blocks(sizes=(8, 8, 8))
    n = W.shape[0]
    W = np.pad(W, ((0, 1), (0, 1)))
    W[n, :8] = 3; W[:8, n] = 3; W[n, 8:16] = 3; W[8:16, n] = 3
    c = cl.consensus(W, n_runs=40, seed=3, resample=False)
    assert c.stability[n] == pytest.approx(1.0)


def test_coassignment_is_one_for_a_pair_that_always_shares_a_cluster():
    runs = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
    C = cl.coassignment(runs)
    assert C[0, 1] == 1.0 and C[0, 2] == pytest.approx(1 / 3)


def test_coassignment_is_nan_for_a_pair_never_active_together():
    runs = np.array([[0, -1, 0], [0, -1, 0]])
    assert np.isnan(cl.coassignment(runs)[0, 1])


# ── brokers and flows ────────────────────────────────────────────────────────

def test_a_country_tied_equally_to_two_clusters_is_the_broker():
    W, _ = blocks(sizes=(8, 8))
    n = W.shape[0]
    W = np.pad(W, ((0, 1), (0, 1)))
    W[n, :8] = 2; W[:8, n] = 2
    W[n, 8:n] = 2; W[8:n, n] = 2                        # (not W[n, n]: no self-loop)
    m = np.append(np.repeat([0, 1], 8), 0)
    P = cl.participation(W, m)
    assert P[n] > 0.45 and P[n] == np.nanmax(P) and np.nanmax(P[:n]) < 0.4
    assert cl.outside_share(W, m)[n] == pytest.approx(0.5, abs=0.02)


def test_a_country_with_all_ties_inside_its_cluster_has_zero_participation():
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = W[2, 3] = W[3, 2] = 3
    m = np.array([0, 0, 1, 1])
    assert np.allclose(cl.participation(W, m), 0)
    assert np.allclose(cl.outside_share(W, m), 0)


def test_flows_add_up_events_between_clusters():
    W = np.zeros((4, 4))
    W[0, 1] = 5; W[0, 2] = 3; W[3, 0] = 2; W[2, 3] = 7
    F = cl.flows(W, np.array([0, 0, 1, 1]))
    assert F.loc[0, 0] == 5 and F.loc[0, 1] == 3 and F.loc[1, 0] == 2 and F.loc[1, 1] == 7


# ── null model ───────────────────────────────────────────────────────────────

def test_the_configuration_null_keeps_every_countrys_events_given_and_received():
    W, _ = blocks()
    rng = np.random.default_rng(1)
    for _ in range(5):
        M = cl.draw_null(W, rng)
        assert M.sum(axis=1) == pytest.approx(W.sum(axis=1)) and M.sum(axis=0) == pytest.approx(W.sum(axis=0))
        assert np.diag(M).sum() == 0


def test_the_null_changes_who_is_tied_to_whom():
    W, _ = blocks()
    assert not np.array_equal(cl.draw_null(W, np.random.default_rng(2)), W)


def test_real_blocks_beat_the_null_model():
    W, _ = blocks()
    c = cl.consensus(W, n_runs=20)
    r = cl.compare_with_null(c, W, cl.null_distribution(W, n_null=100, seed=1))
    assert r["modularity_p"] < 0.05 and r["modularity_z"] > 5


def test_a_hub_artefact_is_not_mistaken_for_blocks():
    """One country is tied to everybody, with huge counts, the rest is noise: the structure is volume, not blocks."""
    rng = np.random.default_rng(4)
    n = 24
    W = rng.poisson(0.6, (n, n)).astype(float)
    np.fill_diagonal(W, 0)
    W[0, :] = rng.poisson(30, n); W[:, 0] = rng.poisson(30, n)
    W[0, 0] = 0
    c = cl.consensus(W, n_runs=20)
    r = cl.compare_with_null(c, W, cl.null_distribution(W, n_null=200, seed=2))
    assert r["modularity_p"] > 0.05


def test_null_distribution_returns_one_row_per_draw():
    W, _ = blocks(sizes=(5, 5))
    nd = cl.null_distribution(W, n_null=10)
    assert nd["modularity"].shape == (10,) and nd["participation"].shape == (10, 10)


# ── hubs and sensitivity ─────────────────────────────────────────────────────

def test_hubs_are_the_countries_with_the_most_events():
    W = np.zeros((4, 4))
    W[2, :] = [5, 5, 0, 5]; W[:, 2] = [4, 4, 0, 4]
    assert cl.hubs(W, 1).tolist() == [2]


def test_removing_a_hub_removes_all_its_ties():
    W, _ = blocks(sizes=(5, 5))
    M = cl.without(W, [0])
    assert M[0].sum() == 0 and M[:, 0].sum() == 0 and M[1:, 1:].sum() == W[1:, 1:].sum()
    assert cl.leiden_once(M)[0] == -1


# ── opponents in opposite blocks? ────────────────────────────────────────────

def test_conflict_between_blocks_gives_a_low_within_share():
    m = np.repeat([0, 1], 10)
    K = np.zeros((20, 20))
    K[:10, 10:] = 3; K[10:, :10] = 3                      # all conflict between the blocks
    r = cl.conflict_between_clusters(m, K, n_perm=300)
    assert r["within_share"] == 0 and r["p_low"] < 0.01 and r["z"] < -5


def test_conflict_inside_blocks_gives_a_high_within_share():
    m = np.repeat([0, 1], 10)
    K = np.zeros((20, 20))
    K[:10, :10] = 2; K[10:, 10:] = 2
    np.fill_diagonal(K, 0)
    r = cl.conflict_between_clusters(m, K, n_perm=300)
    assert r["within_share"] == 1 and r["p_low"] > 0.9


def test_no_conflict_events_gives_nan_not_a_number():
    r = cl.conflict_between_clusters(np.array([0, 0, 1]), np.zeros((3, 3)))
    assert np.isnan(r["within_share"]) and r["events"] == 0


# ── agreement between two partitions ─────────────────────────────────────────

def test_adjusted_rand_of_identical_partitions_is_one_even_with_other_labels():
    assert cl.adjusted_rand(np.array([0, 0, 1, 1, 2]), np.array([7, 7, 3, 3, 5])) == pytest.approx(1.0)


def test_adjusted_rand_matches_a_hand_worked_example():
    assert cl.adjusted_rand(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2])) == pytest.approx(4 / 7)


def test_adjusted_rand_of_unrelated_partitions_is_about_zero():
    rng = np.random.default_rng(0)
    vals = [cl.adjusted_rand(rng.integers(0, 4, 200), rng.integers(0, 4, 200)) for _ in range(20)]
    assert abs(np.mean(vals)) < 0.02


def test_unclustered_countries_are_left_out_of_the_comparison():
    assert cl.adjusted_rand(np.array([0, 0, 1, 1, -1]), np.array([0, 0, 1, 1, 3])) == pytest.approx(1.0)


def test_normalized_mutual_information_bounds():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert cl.normalized_mutual_information(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert cl.normalized_mutual_information(rng.integers(0, 3, 600), rng.integers(0, 3, 600)) < 0.02


# ── following clusters over time ─────────────────────────────────────────────

def test_clusters_keep_their_label_when_most_members_stay():
    p1 = np.array([0, 0, 0, 1, 1, 1])
    p2 = np.array([4, 4, 4, 2, 2, 2])                     # same groups, different labels
    a = cl.align_over_time([p1, p2])
    assert a[1].tolist() == a[0].tolist()


def test_a_cluster_that_splits_keeps_the_label_of_its_bigger_part_and_the_rest_gets_a_new_one():
    p1 = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    p2 = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    a = cl.align_over_time([p1, p2])
    assert a[1][0] == a[0][0] and a[1][5] == a[0][5]
    assert a[1][4] not in (a[0][0], a[0][5])


def test_labels_are_not_reused_for_two_clusters_in_one_period():
    ps = [np.array([0, 0, 1, 1]), np.array([0, 1, 2, 2]), np.array([0, 0, 1, 2])]
    for aligned in cl.align_over_time(ps):
        assert len(set(aligned[aligned >= 0])) == len({tuple(np.flatnonzero(aligned == c)) for c in set(aligned[aligned >= 0])})


# ── drill-down chains ────────────────────────────────────────────────────────

def _chain_graph():
    X = np.zeros((6, 6, 4))
    X[0, 1, 1] = 3; X[1, 2, 1] = 2                         # 0 -> 1 -> 2, all inside cluster 0
    X[2, 3, 2] = 4; X[3, 4, 2] = 5                         # 2 -> 3 -> 4, from cluster 0 into cluster 1
    return X


def test_chains_inside_a_cluster_only_use_its_members():
    m = np.array([0, 0, 0, 1, 1, 1])
    t = cl.chains_between_clusters(_chain_graph(), list("abcdef"), m, 0)
    assert t.path.tolist() == ["a → b → c"] and t.weight.iloc[0] == 2


def test_chains_between_two_clusters_start_in_one_and_end_in_the_other():
    m = np.array([0, 0, 0, 1, 1, 1])
    names = dict(zip("abcdef", ["A", "B", "C", "D", "E", "F"]))
    t = cl.chains_between_clusters(_chain_graph(), list("abcdef"), m, 0, 1, names)
    # C -> D -> E (same window, weight 4) and B -> C -> D (C -> D comes one window after B -> C, weight 2)
    assert t.path.tolist() == ["C → D → E", "B → C → D"] and t.weight.tolist() == [4, 2]
    assert t.next_window.tolist() == [0, 2] and t.same_window.tolist() == [4, 0]


def test_a_graph_without_structure_gives_unstable_clusters():
    rng = np.random.default_rng(5)
    W = rng.poisson(1.0, (24, 24)).astype(float)
    np.fill_diagonal(W, 0)
    c = cl.consensus(W, n_runs=60, seed=1)
    assert np.nanmean(c.stability) < 0.8


def test_countries_without_ties_stay_out_of_the_consensus():
    W, _ = blocks(sizes=(6, 6))
    W = np.pad(W, ((0, 2), (0, 2)))
    c = cl.consensus(W, n_runs=20)
    assert (c.membership[-2:] == -1).all() and np.isnan(c.stability[-2:]).all()


def test_a_higher_resolution_gives_more_and_smaller_clusters():
    """Two big blocks, each made of two tighter sub-blocks: low resolution sees 2, high resolution sees 4."""
    rng = np.random.default_rng(0)
    label = np.repeat(np.arange(4), 8)
    big = label // 2
    n = len(label)
    W = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                W[a, b] = rng.poisson(6 if label[a] == label[b] else (1.2 if big[a] == big[b] else 0.05))
    low = len(set(cl.leiden_once(W, resolution=0.3, seed=1)))
    mid = len(set(cl.leiden_once(W, resolution=1.0, seed=1)))
    high = len(set(cl.leiden_once(W, resolution=3.0, seed=1)))
    assert low <= mid <= high and low < high
    assert low == 2 and high >= 4


def test_outside_share_counts_ties_in_both_directions():
    W = np.zeros((3, 3))
    W[0, 1] = 5                      # 0 gives to 1 (same cluster) ...
    W[2, 0] = 5                      # ... and receives from 2 (other cluster)
    out = cl.outside_share(W, np.array([0, 0, 1]))
    assert out[0] == pytest.approx(0.5)


def test_null_p_counts_draws_equal_to_the_observed_value():
    W, _ = blocks(sizes=(5, 5))
    cons = cl.consensus(W, n_runs=5)
    cons = cl.Consensus(cons.membership, cons.stability, cons.coassign, cons.run_quality, 0.2, cons.modularity, 1.0)
    null = {"modularity": np.array([0.1, 0.2, 0.2, 0.3]), "largest_share": np.array([0.5] * 4),
            "n_clusters": np.array([2] * 4), "participation": np.full((4, 10), 0.1)}
    # three of the four draws are at least as large as 0.2 -> p = (1 + 3) / (1 + 4)
    assert cl.compare_with_null(cons, W, null)["modularity_p"] == pytest.approx(0.8)


def test_a_cluster_with_no_overlap_with_the_earlier_period_gets_a_new_label():
    p1 = np.array([0, 0, 0, 0, -1, -1])
    p2 = np.array([-1, -1, -1, -1, 0, 0])
    a = cl.align_over_time([p1, p2])
    assert a[0][0] != a[1][4]
