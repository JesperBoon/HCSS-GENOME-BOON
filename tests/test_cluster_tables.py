"""Tests for src/cluster_tables.py."""

import numpy as np
import pandas as pd

import cluster_tables as ct

NODES = [10, 20, 30, 40, 50, 60]
NAMES = {10: "Aland", 20: "Bland", 30: "Cland", 40: "Dland", 50: "Eland", 60: "Fland"}


def members(cluster, stability=1.0, participation=0.0, z=0.0, p=1.0, outside=0.0, strength=None):
    n = len(NODES)
    return pd.DataFrame({"country_id": NODES, "cluster": cluster, "stability": stability if np.ndim(stability) else [stability] * n,
                         "participation": participation if np.ndim(participation) else [participation] * n,
                         "participation_z": z if np.ndim(z) else [z] * n, "participation_p": p if np.ndim(p) else [p] * n,
                         "outside_share": outside if np.ndim(outside) else [outside] * n,
                         "strength": strength if strength is not None else [60, 50, 40, 30, 20, 10]})


def test_config_id_matches_the_keys_the_build_script_writes():
    assert ct.config_id("words", "cooperation", "2025Q1", True, 1) == "words|cooperation|2025Q1|log1|hub1|res1.0"
    assert ct.config_id("material", "conflict", "all", False, 0, 0.5) == "material|conflict|all|log0|hub0|res0.5"


def test_blocks_are_named_after_their_busiest_members_and_numbered_from_one():
    m = members([0, 0, 0, 1, 1, -1])
    n = ct.block_names(m, NAMES, top=2)
    assert n == {0: "Block 1: Aland, Bland", 1: "Block 2: Dland, Eland"}


def test_the_same_group_of_countries_gets_the_same_label_on_two_maps():
    ref = np.array([0, 0, 0, 1, 1, 1])
    other = np.array([5, 5, 5, 2, 2, 2])
    a, b = ct.align_pair(ref, other)
    assert a.tolist() == b.tolist()


# ── brokers ──────────────────────────────────────────────────────────────────

def _graph():
    W = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 0), (0, 2), (2, 0), (3, 4), (4, 3), (3, 5), (5, 3)]:
        W[a, b] = 5
    W[2, 3] = W[3, 2] = 4                        # country 2 and 3 tie the blocks together
    return W


def test_a_country_that_ties_two_blocks_is_listed_with_the_other_block():
    m = members([0, 0, 0, 1, 1, 1], participation=[0.0, 0.0, 0.3, 0.3, 0.0, 0.0], z=[0, 0, 3.0, 3.0, 0, 0],
                p=[1, 1, 0.01, 0.01, 1, 1], outside=[0, 0, 0.2, 0.2, 0, 0])
    t = ct.broker_table(_graph(), m, NODES, NAMES)
    assert t.Country.head(2).tolist() == ["Cland", "Dland"]
    assert t.loc[t.Country == "Cland", "Ties mostly to"].iloc[0] == "Block 2"
    assert t["Stands out"].head(2).all() and not t["Stands out"].iloc[2:].any()


def test_unstable_countries_are_not_called_brokers():
    m = members([0, 0, 0, 1, 1, 1], stability=[1, 1, 0.3, 1, 1, 1], participation=[0, 0, 0.9, 0.3, 0, 0])
    assert "Cland" not in ct.broker_table(_graph(), m, NODES, NAMES).Country.tolist()


def test_a_high_participation_without_statistical_support_does_not_stand_out():
    m = members([0, 0, 0, 1, 1, 1], participation=[0, 0, 0.6, 0, 0, 0], z=[0, 0, 1.2, 0, 0, 0], p=[1, 1, 0.3, 1, 1, 1])
    t = ct.broker_table(_graph(), m, NODES, NAMES)
    assert not t["Stands out"].any()


# ── candidate latent ties ────────────────────────────────────────────────────

def test_pairs_in_one_words_block_but_in_different_action_blocks_are_candidates():
    words = np.zeros((6, 6)); words[0, 1] = 7; words[1, 0] = 3; words[0, 2] = 4
    mem_w = np.array([0, 0, 0, 1, 1, 1])
    mem_a = np.array([0, 1, 0, 1, 1, 1])
    stab = np.ones(6)
    t = ct.candidate_latent_pairs(words, mem_w, mem_a, stab, stab, NODES, NAMES)
    assert t.Pair.tolist() == ["Aland – Bland"] and t["Word events"].tolist() == [10]
    assert t["Action blocks"].iloc[0] == "Block 1 / Block 2"


def test_unstable_countries_give_no_candidate_pairs():
    words = np.zeros((6, 6)); words[0, 1] = 7
    stab = np.array([1, 0.4, 1, 1, 1, 1])
    t = ct.candidate_latent_pairs(words, np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 0, 1, 1, 1]), stab, stab, NODES, NAMES)
    assert t.empty


def test_pairs_in_the_same_block_in_both_layers_are_not_candidates():
    words = np.zeros((6, 6)); words[0, 1] = 7
    m = np.array([0, 0, 0, 1, 1, 1])
    assert ct.candidate_latent_pairs(words, m, m.copy(), np.ones(6), np.ones(6), NODES, NAMES).empty


# ── membership over time ─────────────────────────────────────────────────────

def _periods():
    p1 = members([0, 0, 0, 1, 1, 1])
    p2 = members([3, 3, 3, 4, 4, 4])                       # same groups, other labels
    p3 = members([3, 3, 4, 4, 4, -1])                      # country 30 (index 2) moved to the second group, 60 drops out
    return {"2025Q1": p1, "2025Q2": p2, "2025Q3": p3}


def test_a_stable_group_keeps_its_label_across_periods():
    a = ct.membership_over_time(_periods(), NODES)
    assert a["2025Q1"].tolist() == a["2025Q2"].tolist()
    assert a.loc[60, "2025Q3"] == -1


def test_a_country_that_switches_block_is_counted_as_a_change():
    a = ct.membership_over_time(_periods(), NODES)
    c = ct.block_changes(a)
    assert c.loc[30, "changes"] == 1 and c.loc[10, "changes"] == 0
    assert c.loc[60, "comparisons"] == 1                  # only the first pair of periods has both clustered


def test_changes_are_only_counted_where_the_country_is_stable():
    a = ct.membership_over_time(_periods(), NODES)
    stab = pd.DataFrame(1.0, index=NODES, columns=a.columns)
    stab.loc[30, "2025Q3"] = 0.2
    c = ct.block_changes(a, stab)
    assert c.loc[30, "changes"] == 0 and c.loc[30, "comparisons"] == 1


def test_sankey_links_count_countries_that_stay_or_move():
    a = ct.membership_over_time(_periods(), NODES)
    nodes, links = ct.timeline_sankey(a)
    by_pair = {(nodes[s], nodes[t]): n for s, t, n in links}
    first_block = a.loc[10, "2025Q1"]
    assert by_pair[(("2025Q1", first_block), ("2025Q2", a.loc[10, "2025Q2"]))] == 3
    assert sum(n for s, t, n in links if nodes[s][0] == "2025Q2") == 5          # 6 countries, one dropped out in Q3


def test_sankey_leaves_out_unstable_countries_when_asked():
    a = ct.membership_over_time(_periods(), NODES)
    stab = pd.DataFrame(1.0, index=NODES, columns=a.columns)
    stab.loc[[10, 20], "2025Q2"] = 0.1
    _, links = ct.timeline_sankey(a, stab, min_stability=0.7)
    assert sum(n for _, _, n in links) < sum(n for _, _, n in ct.timeline_sankey(a)[1])


def test_a_pair_without_any_words_between_the_two_is_not_a_candidate():
    words = np.zeros((6, 6)); words[0, 1] = 7                  # B and C are in the same words block but never address each other
    mem_w = np.array([0, 0, 0, 1, 1, 1])
    mem_a = np.array([0, 1, 2, 1, 1, 1])
    t = ct.candidate_latent_pairs(words, mem_w, mem_a, np.ones(6), np.ones(6), NODES, NAMES)
    assert t.Pair.tolist() == ["Aland – Bland"]


def test_strip_labels_show_group_number_uncertain_and_no_ties():
    aligned = pd.DataFrame({"Q1": [0, 1, -1], "Q2": [0, 1, 2]}, index=[10, 20, 30])
    stab = pd.DataFrame({"Q1": [0.9, 0.3, np.nan], "Q2": [0.9, 0.9, 0.9]}, index=[10, 20, 30])
    t = ct.strip_labels(aligned, stab, [10, 20, 30], NAMES)
    assert t.loc["Aland"].tolist() == ["1", "1"]
    assert t.loc["Bland"].tolist() == ["?", "2"]                 # block 2 but uncertain in Q1
    assert t.loc["Cland"].tolist() == ["–", "3"]                 # no ties in Q1 is not the same as uncertain


def test_strip_labels_respect_the_requested_countries_and_threshold():
    aligned = pd.DataFrame({"Q1": [0, 1]}, index=[10, 20])
    stab = pd.DataFrame({"Q1": [0.5, 0.5]}, index=[10, 20])
    assert ct.strip_labels(aligned, stab, [20], NAMES, 0.4).iloc[0, 0] == "2"
    assert ct.strip_labels(aligned, stab, [20], NAMES, 0.6).iloc[0, 0] == "?"
