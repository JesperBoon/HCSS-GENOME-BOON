"""Unit tests for src/metrics.py, on small synthetic data whose results can be worked out by hand."""

import math

import numpy as np
import pandas as pd
import pytest

import metrics as mt

ALL_TYPES = ["AGREE", "CONSULT", "SUPPORT", "CONCEDE", "AID", "COOPERATE", "RETREAT",
             "REQUEST", "ACCUSE", "REJECT", "THREATEN", "ASSAULT", "SANCTION", "COERCE",
             "MOBILIZE", "PROTEST"]
FIRST, SECOND = 10, 11   # the two parties of the synthetic conflict


# ── classification ───────────────────────────────────────────────────────────

def test_four_groups_cover_all_16_event_types_exactly_once():
    types = mt.VERBAL_COOP + mt.MATERIAL_COOP + mt.VERBAL_CONF + mt.MATERIAL_CONF
    assert sorted(types) == sorted(ALL_TYPES)


def test_groups_agree_with_the_paper_table():
    """GENOME paper, Table 4: PROTEST and MOBILIZE are material conflict, CONSULT is verbal cooperation."""
    assert "PROTEST" in mt.MATERIAL_CONF and "MOBILIZE" in mt.MATERIAL_CONF
    assert "CONSULT" in mt.VERBAL_COOP and "REQUEST" in mt.VERBAL_CONF


def test_group_dict_and_dyad_lists_describe_the_same_classification():
    assert set(mt.GROUPS["verbal_coop"]) == set(mt.VERBAL_COOP)
    assert set(mt.GROUPS["mat_coop"]) == set(mt.MATERIAL_COOP)
    assert set(mt.GROUPS["verbal_conf"]) == set(mt.VERBAL_CONF)
    assert set(mt.GROUPS["mat_conf"]) == set(mt.MATERIAL_CONF)


def test_every_type_has_a_severity_weight_from_one_of_the_three_bands():
    assert set(mt.TYPE_WEIGHTS) == set(ALL_TYPES)
    assert set(mt.TYPE_WEIGHTS.values()) == {0.15, 0.55, 0.9}
    assert mt.TYPE_WEIGHTS["CONSULT"] < mt.TYPE_WEIGHTS["AGREE"] < mt.TYPE_WEIGHTS["ASSAULT"]


# ── stance and pattern ───────────────────────────────────────────────────────

def test_stance_is_cooperation_minus_conflict_over_total():
    out = mt._stance(pd.Series([3, 0, 2]), pd.Series([1, 3, 2]), min_n=2)
    assert out.tolist() == pytest.approx([0.5, -1.0, 0.0])


def test_stance_is_nan_below_the_minimum_number_of_events():
    out = mt._stance(pd.Series([1, 5]), pd.Series([1, 0]), min_n=3)
    assert math.isnan(out[0]) and out[1] == 1.0


@pytest.mark.parametrize("words, acts, expected", [
    (0.8, 0.6, "Friendly words, friendly actions"),
    (-0.8, -0.6, "Hostile words, hostile actions"),
    (0.8, -0.6, "Friendly words, hostile actions"),
    (-0.8, 0.6, "Hostile words, friendly actions"),
    (0.1, 0.9, "Mixed or neutral"),          # words inside the dead zone
    (0.9, -0.1, "Mixed or neutral"),         # actions inside the dead zone
    (0.2, -0.2, "Friendly words, hostile actions"),   # the boundary itself counts as a stance
])
def test_pattern_names_a_mismatch_only_when_both_signals_are_clear(words, acts, expected):
    assert mt.pattern(words, acts) == expected


# ── side_table (quarterly data) ──────────────────────────────────────────────

def _dyad_file(tmp_path, rows):
    """rows: (actor, target, quarter, {type: n}) -> a dyad_counts.csv with all 16 types present."""
    recs = []
    for actor, target, quarter, counts in rows:
        for t in ALL_TYPES:
            recs.append({"conflict": "A-B", "actor_id": actor, "target_id": target,
                         "quarter": quarter, "key": t, "n": counts.get(t, 0)})
    path = tmp_path / "dyad_counts.csv"
    pd.DataFrame(recs).to_csv(path, index=False)
    return path


@pytest.fixture
def dyads(tmp_path, monkeypatch):
    path = _dyad_file(tmp_path, [
        # actor 1 is friendly in words to FIRST, hostile to SECOND, and hostile in acts to SECOND
        (1, FIRST, "2024Q1", {"CONSULT": 4, "ACCUSE": 1}),
        (1, SECOND, "2024Q1", {"ACCUSE": 5, "ASSAULT": 5}),
        # actor 2 has only two events toward FIRST: too few for a score
        (2, FIRST, "2024Q1", {"AGREE": 2}),
        (2, SECOND, "2024Q1", {"ACCUSE": 3}),
    ])
    monkeypatch.setattr(mt, "DYADS", path)
    return path


def test_side_score_is_stance_toward_second_minus_toward_first(dyads):
    t = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"]).set_index("actor_id")
    assert t.loc[1, "verbal_first"] == pytest.approx(0.6)       # (4 - 1) / 5
    assert t.loc[1, "verbal_second"] == pytest.approx(-1.0)     # only ACCUSE
    assert t.loc[1, "side_verbal"] == pytest.approx(-1.6)       # leans toward FIRST
    assert t.loc[1, "material_second"] == pytest.approx(-1.0)   # only ASSAULT
    assert math.isnan(t.loc[1, "material_first"])               # no material events toward FIRST


def test_a_score_needs_enough_events_toward_both_parties(dyads):
    t = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"], min_n=3).set_index("actor_id")
    assert math.isnan(t.loc[2, "side_verbal"])                  # 2 events toward FIRST < 3
    t2 = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"], min_n=2).set_index("actor_id")
    assert t2.loc[2, "side_verbal"] == pytest.approx(-1.0 - 1.0)  # AGREE only toward FIRST: +1; ACCUSE: -1


def test_event_counts_are_reported_unweighted(dyads):
    t = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"], weighted=True).set_index("actor_id")
    assert t.loc[1, "n_verbal_first"] == 5 and t.loc[1, "n_verbal_second"] == 5


def test_severity_weighting_changes_stance_but_not_the_event_counts(tmp_path, monkeypatch):
    path = _dyad_file(tmp_path, [
        (1, FIRST, "2024Q1", {"AGREE": 4, "ACCUSE": 1}),
        (1, SECOND, "2024Q1", {"ACCUSE": 3, "REQUEST": 1}),
    ])
    monkeypatch.setattr(mt, "DYADS", path)
    plain = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"]).set_index("actor_id")
    heavy = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"], weighted=True).set_index("actor_id")
    assert plain.loc[1, "verbal_first"] == pytest.approx(0.6)                       # (4 - 1) / 5
    expected = (4 * 0.55 - 1 * 0.15) / (4 * 0.55 + 1 * 0.15)                       # AGREE 0.55, ACCUSE 0.15
    assert heavy.loc[1, "verbal_first"] == pytest.approx(expected)
    assert heavy.loc[1, "verbal_first"] > plain.loc[1, "verbal_first"]
    assert heavy.loc[1, "n_verbal_first"] == plain.loc[1, "n_verbal_first"] == 5


def test_only_the_chosen_quarters_are_counted(tmp_path, monkeypatch):
    path = _dyad_file(tmp_path, [
        (1, FIRST, "2024Q1", {"CONSULT": 3}), (1, SECOND, "2024Q1", {"ACCUSE": 3}),
        (1, FIRST, "2024Q2", {"ACCUSE": 3}), (1, SECOND, "2024Q2", {"CONSULT": 3}),
    ])
    monkeypatch.setattr(mt, "DYADS", path)
    q1 = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"]).set_index("actor_id")
    q2 = mt.side_table("A-B", FIRST, SECOND, ["2024Q2"]).set_index("actor_id")
    both = mt.side_table("A-B", FIRST, SECOND, ["2024Q1", "2024Q2"]).set_index("actor_id")
    assert q1.loc[1, "side_verbal"] == pytest.approx(-2.0)
    assert q2.loc[1, "side_verbal"] == pytest.approx(2.0)
    assert both.loc[1, "side_verbal"] == pytest.approx(0.0)


def test_gap_is_acts_minus_words_and_is_centred_on_the_typical_country(tmp_path, monkeypatch):
    path = _dyad_file(tmp_path, [
        (1, FIRST, "2024Q1", {"CONSULT": 3, "AID": 3}), (1, SECOND, "2024Q1", {"ACCUSE": 3, "ASSAULT": 3}),
        (2, FIRST, "2024Q1", {"CONSULT": 3, "AID": 3}), (2, SECOND, "2024Q1", {"ACCUSE": 3, "ASSAULT": 3}),
        (3, FIRST, "2024Q1", {"CONSULT": 3, "ASSAULT": 3}), (3, SECOND, "2024Q1", {"ACCUSE": 3, "AID": 3}),
    ])
    monkeypatch.setattr(mt, "DYADS", path)
    t = mt.side_table("A-B", FIRST, SECOND, ["2024Q1"]).set_index("actor_id")
    assert t.loc[1, "gap"] == pytest.approx(t.loc[1, "side_material"] - t.loc[1, "side_verbal"])
    assert t.gap.median() == pytest.approx(t.loc[1, "gap"])
    assert t.loc[1, "gap_vs_typical"] == pytest.approx(0.0)
    assert t.loc[3, "gap_vs_typical"] != pytest.approx(0.0)


def test_missing_event_types_raise_a_helpful_error(tmp_path, monkeypatch):
    path = tmp_path / "dyad_counts.csv"
    pd.DataFrame([{"conflict": "A-B", "actor_id": 1, "target_id": FIRST, "quarter": "2024Q1",
                   "key": "coop_all", "n": 3}]).to_csv(path, index=False)
    monkeypatch.setattr(mt, "DYADS", path)
    with pytest.raises(RuntimeError, match="build_dyad_counts"):
        mt.side_table("A-B", FIRST, SECOND, ["2024Q1"])


# ── rolling windows (weekly data) ────────────────────────────────────────────

@pytest.fixture
def weekly(tmp_path, monkeypatch):
    rows = [
        # actor 1, four consecutive weeks (Mondays)
        (1, FIRST, "2024-01-01", "CONSULT", 2),
        (1, FIRST, "2024-01-08", "CONSULT", 2),
        (1, SECOND, "2024-01-08", "ACCUSE", 3),
        (1, FIRST, "2024-01-22", "CONSULT", 1),
        (1, FIRST, "2024-01-01", "_done", 0),
    ]
    d = tmp_path / "weekly"
    d.mkdir()
    pd.DataFrame(rows, columns=["actor_id", "target_id", "week_start", "key", "n"]).to_csv(d / "A-B.csv", index=False)
    monkeypatch.setattr(mt, "WEEKLY_DIR", d)
    return d


def test_weekly_loader_drops_the_done_marker(weekly):
    d = mt.load_weekly("A-B")
    assert "_done" not in set(d.key)
    assert mt.weekly_available() == ["A-B"]
    assert mt.weekly_players("A-B") == [1]


def test_rolling_window_pools_the_last_n_weeks(weekly):
    r = mt.rolling_scores("A-B", FIRST, SECOND, window=2, min_n=3).set_index("week_start")
    assert len(r) == 4                                           # 2024-01-01 .. 2024-01-22, one row per week
    assert math.isnan(r.loc["2024-01-01", "side_all"])           # the window is not full yet
    # window = weeks 1+2: toward FIRST 4 cooperative events (+1), toward SECOND 3 hostile (-1)
    assert r.loc["2024-01-08", "all_first"] == pytest.approx(1.0)
    assert r.loc["2024-01-08", "all_second"] == pytest.approx(-1.0)
    assert r.loc["2024-01-08", "side_all"] == pytest.approx(-2.0)
    assert r.loc["2024-01-08", "n_all_first"] == 4 and r.loc["2024-01-08", "n_all_second"] == 3


def test_rolling_window_leaves_a_gap_when_there_are_too_few_events(weekly):
    r = mt.rolling_scores("A-B", FIRST, SECOND, window=2, min_n=3).set_index("week_start")
    assert math.isnan(r.loc["2024-01-15", "side_all"])           # weeks 2+3: only 2 events toward FIRST
    assert math.isnan(r.loc["2024-01-22", "side_all"])


def test_week_end_is_the_sunday_of_the_last_week_in_the_window(weekly):
    r = mt.rolling_scores("A-B", FIRST, SECOND, window=2)
    assert (r.week_end - r.week_start == pd.Timedelta(days=6)).all()
    assert (r.week_end.dt.dayofweek == 6).all()


def test_a_wider_window_can_recover_a_score(weekly):
    r = mt.rolling_scores("A-B", FIRST, SECOND, window=4, min_n=3).set_index("week_start")
    last = r.loc["2024-01-22"]                                   # all four weeks: 5 events toward FIRST, 3 toward SECOND
    assert last.n_all_first == 5 and last.side_all == pytest.approx(-2.0)


# ── map page profile ─────────────────────────────────────────────────────────

def test_country_profile_conflict_share_and_stability_threshold():
    months = [f"2024-{m:02d}" for m in range(1, 7)]
    rows = [dict(country_id=1, month=m, total=10, conflict=c, verbal_coop=0, mat_coop=0,
                 verbal_conf=0, mat_conf=0) for m, c in zip(months, [2, 4, 6, 8, 5, 5])]
    rows += [dict(country_id=2, month=m, total=10, conflict=5, verbal_coop=0, mat_coop=0,
                  verbal_conf=0, mat_conf=0) for m in months[:2]]       # only two months of data
    monthly = pd.DataFrame(rows)
    p = mt.country_profile(monthly, "2024-01", "2024-06", min_events=10).set_index("country_id")
    assert p.loc[1, "conflict_share"] == pytest.approx(30 / 60)
    assert p.loc[1, "stability_cv"] > 0
    assert np.isnan(p.loc[2, "stability_cv"])                          # fewer than 4 months: no stability
