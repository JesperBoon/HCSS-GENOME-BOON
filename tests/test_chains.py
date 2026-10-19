"""Tests for src/chains.py on tiny graphs whose chains can be counted by hand."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import chains as ch

A, B, C, D = 0, 1, 2, 3


def graph(n=4, T=6):
    return np.zeros((n, n, T))


# ── chain weights ────────────────────────────────────────────────────────────

def test_a_chain_in_the_same_window_is_as_strong_as_its_weakest_link():
    X = graph()
    X[A, B, 2], X[B, C, 2] = 3, 5
    same, nxt = ch.chain_weights(X, X)
    assert same[A, B, C] == 3 and nxt[A, B, C] == 0


def test_a_chain_in_the_next_window_is_counted_as_ordered():
    X = graph()
    X[A, B, 2], X[B, C, 3] = 3, 5
    same, nxt = ch.chain_weights(X, X)
    assert same[A, B, C] == 0 and nxt[A, B, C] == 3


def test_the_product_is_available_as_an_alternative_weight():
    X = graph()
    X[A, B, 2], X[B, C, 2] = 3, 5
    same, _ = ch.chain_weights(X, X, combine="product")
    assert same[A, B, C] == 15


def test_support_passed_on_before_it_was_received_is_not_a_chain():
    X = graph()
    X[A, B, 3], X[B, C, 2] = 4, 4                  # B gives to C BEFORE A gives to B
    same, nxt = ch.chain_weights(X, X)
    assert same[A, B, C] == 0 and nxt[A, B, C] == 0


def test_a_window_two_steps_later_does_not_count_as_next():
    X = graph()
    X[A, B, 1], X[B, C, 3] = 4, 4
    same, nxt = ch.chain_weights(X, X)
    assert same[A, B, C] == 0 and nxt[A, B, C] == 0


def test_a_chain_that_returns_to_its_start_is_removed():
    X = graph()
    X[A, B, 2], X[B, A, 2] = 3, 3
    same, nxt = ch.chain_weights(X, X)
    assert same[A, B, A] == 0 and nxt[A, B, A] == 0


def test_weights_add_up_over_windows():
    X = graph()
    X[A, B, 1], X[B, C, 1] = 2, 2
    X[A, B, 4], X[B, C, 4] = 1, 4
    same, _ = ch.chain_weights(X, X)
    assert same[A, B, C] == 3


def test_chains_can_join_two_different_layers():
    support, attacks = graph(), graph()
    support[A, B, 2] = 4
    attacks[B, D, 2] = 6
    same, _ = ch.chain_weights(support, attacks)
    assert same[A, B, D] == 4


# ── null model ───────────────────────────────────────────────────────────────

def test_shuffling_keeps_every_edge_total_but_moves_the_timing():
    rng = np.random.default_rng(1)
    X = graph(T=20)
    X[A, B, 3], X[A, B, 11] = 5, 7
    Xp = ch.shuffle_edges(X, rng)
    assert Xp[A, B].sum() == 12 and sorted(Xp[A, B][Xp[A, B] > 0]) == [5, 7]
    assert not np.array_equal(Xp, X)


def test_a_steady_flow_through_a_hub_is_pure_volume_and_not_significant():
    X = graph(T=12)
    X[A, B, :], X[B, C, :] = 10, 10                # busy every single window
    res = ch.chain_significance(X, X, n_perm=200)
    assert res["weight"][A, B, C] == 12 * 10 + 11 * 10
    assert res["p"][A, B, C] == pytest.approx(1.0)
    assert res["z"][A, B, C] == 0


def test_busy_edges_that_overlap_by_chance_are_not_significant():
    rng = np.random.default_rng(3)
    X = graph(T=24)
    X[A, B, :], X[B, C, :] = rng.poisson(20, 24), rng.poisson(20, 24)
    res = ch.chain_significance(X, X, n_perm=300, seed=5)
    assert res["p"][A, B, C] > 0.05


ACTIVE = (2, 7, 12, 17, 22, 27)          # six windows out of 36, spaced so they never touch


def coordinated(windows=ACTIVE, T=36):
    X = graph(T=T)
    for t in windows:
        X[A, B, t], X[B, C, t] = 10, 10
    return X


def test_edges_that_are_active_together_in_several_windows_are_significant():
    X = coordinated()
    res = ch.chain_significance(X, X, n_perm=1000, seed=2, null="shuffle")
    assert res["weight"][A, B, C] == 60
    assert res["p"][A, B, C] < 0.01
    assert res["z"][A, B, C] > 3
    assert res["expected"][A, B, C] < 30


def test_more_coordination_gives_a_smaller_p_value():
    few = ch.chain_significance(coordinated(ACTIVE[:3]), coordinated(ACTIVE[:3]), n_perm=1000, seed=2, null="shuffle")
    many = ch.chain_significance(coordinated(), coordinated(), n_perm=1000, seed=2, null="shuffle")
    assert many["p"][A, B, C] < few["p"][A, B, C]


def test_the_null_model_shuffles_both_sides_when_given_two_graphs():
    X, Y = graph(T=36), graph(T=36)
    for t in ACTIVE:
        X[A, B, t], Y[B, D, t] = 10, 10
    res = ch.chain_significance(X, Y, n_perm=1000, seed=2, null="shuffle")
    assert res["p"][A, B, D] < 0.01


def test_chains_table_names_sorts_and_filters():
    X = coordinated()
    X[B, D, 5], X[A, B, 5] = 1, 1
    res = ch.chain_significance(X, X, n_perm=1000, null="shuffle")
    names = {A: "Aland", B: "Bland", C: "Cland", D: "Dland"}
    t = ch.chains_table(res, [A, B, C, D], names, min_weight=1)
    assert t.iloc[0].chain == "Aland → Bland → Cland"
    only_sig = ch.chains_table(res, [A, B, C, D], names, min_weight=1, p_max=0.01)
    assert list(only_sig.chain) == ["Aland → Bland → Cland"]


# ── brokers and cycles ───────────────────────────────────────────────────────

def test_a_broker_both_receives_and_gives_in_the_same_window():
    X = graph()
    X[A, B, 1], X[B, C, 1] = 3, 2
    score = ch.broker_scores(X)
    assert score[B] == 2                           # min(3 received, 2 given)
    assert score[A] == 0 and score[C] == 0         # only gives / only receives


def test_a_country_that_receives_in_one_window_and_gives_in_another_is_no_broker_in_this_score():
    X = graph()
    X[A, B, 1], X[B, C, 4] = 3, 3
    assert ch.broker_scores(X)[B] == 0


def test_a_three_country_cycle_is_found_once_in_a_fixed_direction():
    X = graph()
    X[A, B, 2], X[B, C, 2], X[C, A, 3] = 1, 1, 1
    w = ch.cycle_weights(X)
    assert w[A, B, C] == 1 and w.sum() == 1


def test_no_cycle_without_a_closing_edge():
    X = graph()
    X[A, B, 2], X[B, C, 2] = 1, 1
    assert ch.cycle_weights(X).sum() == 0


def test_cycle_significance_returns_weights_and_p_values():
    X = graph(T=36)
    for t in ACTIVE:
        X[A, B, t] = X[B, C, t] = X[C, A, t] = 3
    res = ch.cycle_significance(X, n_perm=300)
    assert res["weight"][A, B, C] > 0 and res["p"][A, B, C] < 0.05


# ── building the array from the stored files ─────────────────────────────────

def _frames():
    events = pd.DataFrame([
        {"event_id": "e1", "event_date": "2024-01-02", "event_type": "AID", "category": "COOPERATION"},
        {"event_id": "e2", "event_date": "2024-01-20", "event_type": "AID", "category": "COOPERATION"},
        {"event_id": "e3", "event_date": "2024-01-03", "event_type": "ASSAULT", "category": "CONFLICT"},
        {"event_id": "e4", "event_date": "2024-01-04", "event_type": "CONSULT", "category": "COOPERATION"},
    ])
    edges = pd.DataFrame([("e1", 10, 20), ("e2", 10, 20), ("e3", 10, 20), ("e4", 10, 20), ("e1", 99, 20)],
                         columns=["event_id", "actor_id", "recipient_id"])
    leaves = pd.DataFrame([
        {"event_type": "ASSAULT", "actor_id": 10, "recipient_id": 20, "window_start": "2024-01-01",
         "window_end": "2024-01-14", "count": 14, "source": "count"},
        {"event_type": "AID", "actor_id": 10, "recipient_id": 20, "window_start": "2024-01-01",
         "window_end": "2024-01-14", "count": 1, "source": "events"}])
    return events, edges, leaves


def test_events_are_binned_into_windows_by_their_date():
    ev, ed, lv = _frames()
    g = ch.build_graph(ev, ed, lv, [10, 20], date(2024, 1, 1), date(2024, 2, 25), 2, "material", "cooperation")
    assert g.counts.shape == (2, 2, 4)                              # 56 days / 14
    assert g.counts[0, 1, 0] == 1 and g.counts[0, 1, 1] == 1        # 2 Jan in window 0, 20 Jan in window 1
    assert g.counts.sum() == 2                                      # CONSULT is words, ASSAULT is conflict: left out


def test_layer_and_category_select_the_event_types():
    ev, ed, lv = _frames()
    words = ch.build_graph(ev, ed, lv, [10, 20], date(2024, 1, 1), date(2024, 2, 25), 2, "words", "cooperation")
    assert words.counts[0, 1].sum() == 1                            # only CONSULT


def test_countries_outside_the_node_list_are_ignored():
    ev, ed, lv = _frames()
    g = ch.build_graph(ev, ed, lv, [10, 20], date(2024, 1, 1), date(2024, 2, 25), 2, "material", "cooperation")
    assert g.counts.sum() == 2                                      # the edge from country 99 is dropped


def test_windows_known_only_as_a_count_are_spread_evenly_over_their_days():
    ev, ed, lv = _frames()
    g = ch.build_graph(ev, ed, lv, [10, 20], date(2024, 1, 1), date(2024, 2, 25), 1, "material", "conflict",
                       event_types=["ASSAULT"])
    # e3 on 3 Jan (window 0) plus a count-only leaf of 14 events over 14 days = 7 per week
    assert g.counts[0, 1, 0] == pytest.approx(1 + 7)
    assert g.counts[0, 1, 1] == pytest.approx(7)


def test_events_outside_the_period_are_dropped_not_wrapped_around():
    ev, ed, lv = _frames()
    extra = pd.DataFrame([
        {"event_id": "old", "event_date": "2023-12-20", "event_type": "AID", "category": "COOPERATION"},
        {"event_id": "late", "event_date": "2024-12-01", "event_type": "AID", "category": "COOPERATION"}])
    ev = pd.concat([ev, extra])
    ed = pd.concat([ed, pd.DataFrame([("old", 10, 20), ("late", 10, 20)],
                                     columns=["event_id", "actor_id", "recipient_id"])])
    g = ch.build_graph(ev, ed, lv, [10, 20], date(2024, 1, 1), date(2024, 2, 25), 2, "material", "cooperation")
    assert g.counts.sum() == 2                      # the same two events as before; neither stray event counted
    assert g.counts[0, 1, -1] == 0                  # a negative index would have put "old" in the last window


# ── false discovery rate and refined p-values ────────────────────────────────

def test_benjamini_hochberg_matches_a_hand_worked_example():
    q = ch.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    # sorted p: .01 .03 .04 .20 -> times m/rank: .04 .06 .0533 .20 -> running minimum from the end: .04 .0533 .0533 .20
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20])


def test_q_values_are_never_below_p_and_never_above_one():
    p = np.random.default_rng(0).random(500) ** 3
    q = ch.benjamini_hochberg(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()


def test_many_chance_findings_do_not_survive_the_correction():
    p = np.random.default_rng(1).random(2000)        # 2000 tests, no real effect at all
    assert (p < 0.05).sum() > 50                      # plenty of "significant" results by luck...
    assert (ch.benjamini_hochberg(p) <= 0.1).sum() == 0   # ...none survive the correction


def test_a_real_effect_survives_the_correction():
    p = np.concatenate([np.full(5, 1e-6), np.random.default_rng(2).random(995)])
    assert (ch.benjamini_hochberg(p) <= 0.05)[:5].all()


def test_refined_p_values_agree_with_the_full_test_and_can_go_lower():
    X = coordinated()
    full = ch.chain_significance(X, X, n_perm=1000, seed=2, null="shuffle")["p"][A, B, C]
    refined = ch.refine_p_values(X, X, np.array([[A, B, C]]), n_perm=5000, seed=3, null="shuffle")[0]
    assert refined < 0.01 and abs(refined - full) < 0.01
    assert refined >= 1 / 5001


def test_refined_p_values_for_steady_flows_stay_at_one():
    X = graph(T=12)
    X[A, B, :], X[B, C, :] = 10, 10
    assert ch.refine_p_values(X, X, np.array([[A, B, C]]), n_perm=300)[0] == pytest.approx(1.0)


def test_refined_p_values_work_across_two_graphs():
    X, Y = graph(T=36), graph(T=36)
    for t in ACTIVE:
        X[A, B, t], Y[B, D, t] = 10, 10
    assert ch.refine_p_values(X, Y, np.array([[A, B, D]]), n_perm=2000, seed=4, null="shuffle")[0] < 0.01


# ── the matched null: busy weeks that are busy for everyone ─────────────────

def war_graph(T=36, n=10):
    """Quiet noise everywhere, plus a six-window 'war' in which many edges including A->B and B->C are busy."""
    rng = np.random.default_rng(7)
    X = np.zeros((n, n, T))
    for a in range(4, n):                          # background edges that have nothing to do with A, B, C
        for b in range(4, n):
            if a != b:
                X[a, b, :] = rng.poisson(0.3, T)
                X[a, b, 14:20] += rng.poisson(8, 6)
    X[A, B, 14:20] += 5
    X[B, C, 14:20] += 5
    return X


def test_activity_order_lists_the_windows_from_quiet_to_busy():
    X = war_graph()
    order = ch.activity_order([X])
    assert sorted(order) == list(range(36))
    assert set(order[-6:]) == set(range(14, 20))     # the six war windows are the busiest


def test_a_matched_shuffle_keeps_totals_and_keeps_busy_events_in_busy_windows():
    X = war_graph()
    order = ch.activity_order([X])
    Xp = ch.shuffle_edges(X, np.random.default_rng(0), order, tolerance=1.0)
    assert Xp.sum() == pytest.approx(X.sum())
    assert Xp[A, B].sum() == pytest.approx(X[A, B].sum())
    in_war = lambda arr: arr[..., 14:20].sum() / arr.sum()
    assert in_war(Xp[A, B][None, :]) > 0.7           # nearly all of A->B stays in the war windows...
    plain = ch.shuffle_edges(X, np.random.default_rng(0), None)
    assert in_war(plain[A, B][None, :]) < 0.5        # ...whereas a plain shuffle scatters it


def test_tolerance_zero_changes_nothing_and_the_default_scales_with_the_number_of_windows():
    X = war_graph()
    order = ch.activity_order([X])
    assert np.array_equal(ch.shuffle_edges(X, np.random.default_rng(0), order, tolerance=0.0), X)
    assert ch._tol(None, 37) == pytest.approx(0.08 * 37) and ch._tol(2.0, 37) == 2.0


def test_a_shared_war_period_looks_like_coordination_under_a_plain_shuffle_but_not_a_matched_one():
    X = war_graph()
    plain = ch.chain_significance(X, X, n_perm=500, seed=1, null="shuffle")["p"][A, B, C]
    matched = ch.chain_significance(X, X, n_perm=500, seed=1, null="matched", tolerance=2.0)["p"][A, B, C]
    assert plain < 0.01                              # fooled by the shared busy weeks
    assert matched > 0.1                             # largely explained by overall activity
    assert matched > 20 * plain


def test_an_unknown_null_model_is_refused():
    X = graph()
    with pytest.raises(ValueError):
        ch.chain_significance(X, X, n_perm=2, null="magic")


# ── reach: who stands behind a country ───────────────────────────────────────

def _pairs(rows):
    return pd.DataFrame(rows, columns=["actor", "recipient", "weight"])


def test_upstream_reach_follows_support_backwards_in_steps():
    p = _pairs([("Z", "A", 2), ("A", "F", 5), ("Q", "F", 1), ("F", "R", 4)])
    up = ch.reach(p, "F", depth=2, direction="upstream").sort_values(["step", "actor"])
    assert list(zip(up.step, up.actor, up.recipient)) == [(1, "A", "F"), (1, "Q", "F"), (2, "Z", "A")]


def test_downstream_reach_follows_support_forwards():
    p = _pairs([("F", "R", 4), ("R", "S", 3), ("A", "F", 5)])
    down = ch.reach(p, "F", depth=2, direction="downstream")
    assert list(zip(down.step, down.actor, down.recipient)) == [(1, "F", "R"), (2, "R", "S")]


def test_depth_one_stops_after_the_first_step():
    p = _pairs([("Z", "A", 2), ("A", "F", 5)])
    assert set(ch.reach(p, "F", depth=1).step) == {1}


def test_loops_do_not_repeat_countries():
    p = _pairs([("A", "F", 1), ("F", "A", 1), ("B", "A", 1), ("A", "B", 1)])
    up = ch.reach(p, "F", depth=3)
    assert not any(up.actor == "F") and len(up) == 2               # A -> F, then B -> A; nothing goes round again


def test_a_country_nobody_supports_has_an_empty_upstream():
    assert ch.reach(_pairs([("F", "R", 1)]), "F").empty


def test_reach_rejects_an_unknown_direction():
    with pytest.raises(ValueError):
        ch.reach(_pairs([("A", "F", 1)]), "F", direction="sideways")


# ── pair weights ─────────────────────────────────────────────────────────────

def test_pair_weights_count_events_per_ordered_pair_in_the_period_and_types():
    ev, ed, _ = _frames()
    w = ch.pair_weights(ev, ed, ["AID"], since="2024-01-01", until="2024-01-31")
    assert w.to_dict("records") == [{"actor": 10, "recipient": 20, "weight": 2}, {"actor": 99, "recipient": 20, "weight": 1}]
    only_late = ch.pair_weights(ev, ed, ["AID"], since="2024-01-15")
    assert only_late.to_dict("records") == [{"actor": 10, "recipient": 20, "weight": 1}]


def test_pair_weights_end_date_is_inclusive_and_other_types_are_left_out():
    ev, ed, _ = _frames()
    assert ch.pair_weights(ev, ed, ["AID"], until="2024-01-02").weight.sum() == 2         # e1 on 2 Jan, two edges
    assert ch.pair_weights(ev, ed, ["ASSAULT"]).weight.tolist() == [1]
    assert ch.pair_weights(ev, ed, ["AID"], since="2025-01-01").empty
