"""Tests for src/edge_counts.py on synthetic events with known dates."""

from datetime import date, timedelta

import pytest

import edge_counts as ec
import taxonomy as tx
from edge_counts import Window, adaptive_windows, compact_event, split

WHOLE = Window(date(2024, 1, 1), date(2024, 12, 31))


def counter(days):
    """count_fn over a list of event dates; also records how often it was called."""
    calls = []

    def count(w):
        calls.append(w)
        return sum(1 for d in days if w.start <= d <= w.end)
    count.calls = calls
    return count


def days_spread(n, start=date(2024, 1, 1), span=360):
    return [start + timedelta(days=(i * span) // n) for i in range(n)]


# ── window helpers ───────────────────────────────────────────────────────────

def test_quarters_are_clipped_to_the_window():
    qs = split(Window(date(2024, 1, 1), date(2024, 8, 10)), "quarter")
    assert qs == [Window(date(2024, 1, 1), date(2024, 3, 31)), Window(date(2024, 4, 1), date(2024, 6, 30)),
                  Window(date(2024, 7, 1), date(2024, 8, 10))]


def test_a_window_inside_one_quarter_gives_that_window_back():
    w = Window(date(2024, 2, 10), date(2024, 2, 20))
    assert split(w, "quarter") == [w]


def test_weeks_run_monday_to_sunday_and_are_clipped():
    ws = split(Window(date(2024, 2, 1), date(2024, 2, 29)), "week")      # 1 Feb 2024 is a Thursday
    assert ws[0] == Window(date(2024, 2, 1), date(2024, 2, 4))
    assert ws[1].start.weekday() == 0 and ws[1].end.weekday() == 6
    assert ws[-1] == Window(date(2024, 2, 26), date(2024, 2, 29))


@pytest.mark.parametrize("level", ["quarter", "month", "week"])
def test_children_cover_the_parent_without_gaps_or_overlap(level):
    ws = split(Window(date(2024, 1, 17), date(2024, 11, 3)), level)
    assert ws[0].start == date(2024, 1, 17) and ws[-1].end == date(2024, 11, 3)
    for a, b in zip(ws, ws[1:]):
        assert b.start == a.end + timedelta(days=1)


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        split(WHOLE, "decade")


# ── adaptive splitting ───────────────────────────────────────────────────────

def test_a_cell_with_few_events_is_closed_at_once_with_one_count_query():
    count = counter(days_spread(40))
    leaves = adaptive_windows(count, WHOLE)
    assert len(leaves) == 1 and leaves[0].window == WHOLE
    assert leaves[0].total == 40 and leaves[0].has_events
    assert len(count.calls) == 1


def test_an_empty_cell_gives_no_leaves():
    assert adaptive_windows(counter([]), WHOLE) == []


def test_a_known_total_saves_the_first_query():
    count = counter(days_spread(40))
    adaptive_windows(count, WHOLE, known_total=40)
    assert count.calls == []


def test_a_busy_cell_is_split_until_every_window_has_at_most_250_events():
    days = days_spread(900)
    leaves = adaptive_windows(counter(days), WHOLE, finest="week")
    assert all(leaf.total <= ec.LIMIT and leaf.has_events for leaf in leaves)
    assert sum(leaf.total for leaf in leaves) == 900                     # nothing lost, nothing counted twice


def test_leaves_are_disjoint_and_hold_exactly_the_events_inside_them():
    days = days_spread(900)
    leaves = adaptive_windows(counter(days), WHOLE)
    spans = sorted((leaf.window.start, leaf.window.end) for leaf in leaves)
    for (_, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 > e1
    for leaf in leaves:
        assert leaf.total == sum(1 for d in days if leaf.window.start <= d <= leaf.window.end)


def test_windows_with_no_events_are_not_split_further():
    days = [date(2024, 1, 1) + timedelta(days=i % 20) for i in range(600)]          # all in January
    count = counter(days)
    leaves = adaptive_windows(count, WHOLE, finest="week")
    assert all(leaf.window.end <= date(2024, 3, 31) for leaf in leaves)
    # 1 whole + 4 quarters + 3 months of Q1 + weeks of the busy month, nowhere near 52 weekly queries
    assert len(count.calls) < 25


def test_the_finest_level_stops_the_splitting_and_marks_the_leaf_as_count_only():
    days = [date(2024, 1, 15)] * 400                                                   # 400 events on one day
    leaves = adaptive_windows(counter(days), WHOLE, finest="month")
    assert len(leaves) == 1
    assert leaves[0].total == 400 and not leaves[0].has_events
    assert leaves[0].window == Window(date(2024, 1, 1), date(2024, 1, 31))


def test_finest_must_be_a_real_level():
    with pytest.raises(ValueError):
        adaptive_windows(counter([]), WHOLE, finest="whole")


# ── compact events ───────────────────────────────────────────────────────────

def _event(actors, recipients, **kw):
    base = {"id": "abc", "event_date": "2025-03-04", "event_type": "AID", "category": "COOPERATION",
            "intensity": 0.8, "article_count": 3,
            "actor": {"countries": [{"id": i, "name": f"c{i}"} for i in actors]},
            "recipient": {"countries": [{"id": i, "name": f"c{i}"} for i in recipients]},
            "event_summary": "long text", "source_quote": "long quote"}
    base.update(kw)
    return base


def test_compact_event_keeps_only_the_small_fields():
    row, _ = compact_event(_event([1], [2]), {1, 2})
    assert set(row) == {"event_id", "event_date", "event_type", "category", "intensity", "article_count"}


def test_one_edge_per_actor_recipient_combination_inside_the_country_set():
    _, edges = compact_event(_event([1, 3], [2, 4]), {1, 2, 3})
    assert edges == [("abc", 1, 2), ("abc", 3, 2)]                       # 4 is outside the set


def test_an_actor_that_is_also_a_recipient_makes_no_self_edge():
    _, edges = compact_event(_event([1, 2], [2]), {1, 2})
    assert edges == [("abc", 1, 2)]


def test_organisations_without_an_id_and_events_without_a_recipient_make_no_edges():
    ev = _event([1], [])
    ev["recipient"]["countries"] = [{"id": None, "name": "European Union"}]
    assert compact_event(ev, {1})[1] == []


# ── the single taxonomy ──────────────────────────────────────────────────────

def test_taxonomy_follows_table_4_of_the_paper_and_covers_16_types_once():
    assert sorted(tx.ALL_TYPES) == sorted(set(tx.ALL_TYPES)) and len(tx.ALL_TYPES) == 16
    assert tx.layer_of("AID") == "material" and tx.category_of("AID") == "cooperation"
    assert tx.layer_of("PROTEST") == "material" and tx.category_of("PROTEST") == "conflict"
    assert tx.layer_of("CONSULT") == "words" and tx.category_of("REQUEST") == "conflict"


def test_exactly_250_events_is_closed_but_251_is_split():
    day = date(2024, 5, 20)
    closed = adaptive_windows(counter([day] * 250), WHOLE)
    assert len(closed) == 1 and closed[0].has_events and closed[0].total == 250
    split_up = adaptive_windows(counter([day] * 251), WHOLE, finest="week")
    assert len(split_up) == 1 and split_up[0].window.start <= day <= split_up[0].window.end
    assert split_up[0].window != WHOLE and split_up[0].total == 251 and not split_up[0].has_events
