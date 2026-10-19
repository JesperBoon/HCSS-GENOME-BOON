"""Adaptive time windows for free count queries, and the compact form of an event.

A count query returns an exact `total`; page 1 of the same query returns the events themselves
when `total` is at most 250 (verified in scripts/probe_chains.py). So for one cell
(actor country -> recipient country, event type) we start with the whole period and only split
into quarters, months and weeks while a window holds more than 250 events. A window with at most
250 events is "closed": its events are fetched from page 1, which also gives exact counts at any
finer resolution for free.

All functions here are pure (no network): the caller supplies `count_fn(window) -> int`.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

LIMIT = 250                                   # events on the free page 1
LEVELS = ("whole", "quarter", "month", "week")


@dataclass(frozen=True)
class Window:
    start: date
    end: date                                 # inclusive, like the API's event_date_to


@dataclass(frozen=True)
class Leaf:
    """A window where splitting stopped."""
    window: Window
    total: int
    has_events: bool                          # True: total <= LIMIT, so page 1 holds every event


def _clip(a: date, b: date, window: Window):
    start, end = max(a, window.start), min(b, window.end)
    return Window(start, end) if start <= end else None


def _periods(window: Window, freq: str) -> list[Window]:
    out = []
    for p in pd.period_range(window.start, window.end, freq=freq):
        w = _clip(p.start_time.date(), p.end_time.date(), window)
        if w:
            out.append(w)
    return out


def _weeks(window: Window) -> list[Window]:
    """Monday-to-Sunday weeks, clipped to the window."""
    out, start = [], window.start
    while start <= window.end:
        sunday = start + timedelta(days=6 - start.weekday())
        out.append(Window(start, min(sunday, window.end)))
        start = sunday + timedelta(days=1)
    return out


def split(window: Window, level: str) -> list[Window]:
    """Child windows of `window` at `level` ('quarter', 'month' or 'week'), clipped to it."""
    if level == "quarter":
        return _periods(window, "Q")
    if level == "month":
        return _periods(window, "M")
    if level == "week":
        return _weeks(window)
    raise ValueError(f"unknown level {level!r}")


def adaptive_windows(count_fn: Callable[[Window], int], whole: Window, finest: str = "week",
                     limit: int = LIMIT, known_total: int | None = None) -> list[Leaf]:
    """Split `whole` until every non-empty window has <= limit events, or the `finest` level is reached.

    Empty windows are dropped. The returned leaves are disjoint and their totals add up to the
    total of `whole`. A leaf with has_events=False holds more than `limit` events and is
    available as a count only.
    """
    if finest not in LEVELS[1:]:
        raise ValueError(f"finest must be one of {LEVELS[1:]}")
    stop_at = LEVELS.index(finest)
    leaves: list[Leaf] = []

    def visit(window: Window, depth: int, total: int) -> None:
        if total == 0:
            return
        if total <= limit:
            leaves.append(Leaf(window, total, True))
        elif depth >= stop_at:
            leaves.append(Leaf(window, total, False))
        else:
            for child in split(window, LEVELS[depth + 1]):
                visit(child, depth + 1, count_fn(child))

    visit(whole, 0, count_fn(whole) if known_total is None else known_total)
    return leaves


def compact_event(event: dict, country_ids: set[int]) -> tuple[dict, list[tuple]]:
    """One API event -> a small row, plus its (actor, recipient) edges inside `country_ids`.

    An event with several actor or recipient countries gives one edge per combination;
    an actor that is also the recipient gives none. Countries without an id (organisations) are ignored.
    """
    row = {"event_id": event["id"], "event_date": event["event_date"], "event_type": event["event_type"],
           "category": event["category"], "intensity": event["intensity"],
           "article_count": event["article_count"]}
    actors = {c["id"] for c in event["actor"]["countries"] if c.get("id") in country_ids}
    recipients = {c["id"] for c in event["recipient"]["countries"] if c.get("id") in country_ids}
    edges = [(event["id"], a, r) for a in sorted(actors) for r in sorted(recipients) if a != r]
    return row, edges
