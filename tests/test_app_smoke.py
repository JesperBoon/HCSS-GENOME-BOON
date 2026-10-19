"""Smoke test: run every page of the Streamlit app on the real data files and check for exceptions.

Skipped when the processed data has not been built yet (see README, 'Build the data').
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow

if not (ROOT / "data" / "processed" / "dyad_counts.csv").exists():
    pytest.skip("processed data not built", allow_module_level=True)

APP = str(ROOT / "app.py")


def _run():
    return AppTest.from_file(APP, default_timeout=300).run()


def _errors(at):
    return [e.value[:300] for e in at.exception]


def test_every_page_renders():
    at = _run()
    assert not _errors(at)
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not _errors(at), page


def test_every_conflict_renders_per_section():
    """Each section on page 1 has its own conflict selectbox (B, C, D). Cycle through conflicts."""
    at = _run()
    # Page 1 has three conflict selectors (sections B, C, D); selectbox[0] is section B
    for conflict in at.selectbox[0].options:
        at.selectbox[0].select(conflict).run()
        assert not _errors(at), conflict


def test_page1_renders_without_errors():
    """Page 1 should render cleanly — no weighting toggle or window slider any more."""
    at = _run()
    assert not _errors(at)


# ── support chains (pages 3 and 4) ───────────────────────────────────────────

NEEDS_EDGES = pytest.mark.skipif(not (ROOT / "data" / "processed" / "edges" / "events.csv").exists(),
                                 reason="edge data not built (scripts/build_edge_counts.py)")
NEEDS_RESULTS = pytest.mark.skipif(not (ROOT / "data" / "processed" / "chains" / "chains.csv").exists(),
                                   reason="chain results not built (scripts/build_chain_results.py)")


def _go(at, page_prefix):
    page = next(p for p in at.sidebar.radio[0].options if p.startswith(page_prefix))
    at.sidebar.radio[0].set_value(page).run()
    assert not _errors(at), page
    return at


@NEEDS_EDGES
def test_chain_map_controls_all_render():
    at = _go(_run(), "4")
    for radio in [r for r in at.radio if r.label != "Navigation"]:
        for option in radio.options:
            radio.set_value(option)
            at.run()
            assert not _errors(at), (radio.label, option)


@NEEDS_EDGES
@NEEDS_RESULTS
def test_chain_statistics_and_word_test_controls_all_render():
    at = _go(_run(), "4")
    assert [t for t in at.toggle if "stand out" in t.label], "statistics section missing"
    for radio in [r for r in at.radio if r.label != "Navigation"]:
        for option in radio.options:
            radio.set_value(option)
            at.run()
            assert not _errors(at), (radio.label, option)


@NEEDS_EDGES
def test_every_case_and_party_renders_and_shows_its_events():
    at = _go(_run(), "5")
    for case in at.selectbox[0].options:
        at.selectbox[0].select(case).run()
        for party in [r for r in at.radio if r.label.startswith(("Follow", "Focus"))][0].options:
            [r for r in at.radio if r.label.startswith(("Follow", "Focus"))][0].set_value(party).run()
            assert not _errors(at), (case, party)


# ── support clusters (page 2) ────────────────────────────────────────────────

NEEDS_CLUSTERS = pytest.mark.skipif(not (ROOT / "data" / "processed" / "clusters" / "configs.csv").exists(),
                                    reason="cluster results not built (scripts/build_cluster_results.py)")


def _widget(at, kind, label, nth=0):
    found = [w for w in getattr(at, kind) if w.label == label]
    return found[nth] if len(found) > nth else None


@NEEDS_EDGES
@NEEDS_CLUSTERS
def test_cluster_page_every_choice_renders():
    at = _go(_run(), "2")
    for label in ("Show", "Kind", "Counting", "Biggest countries"):
        first = _widget(at, "radio", label).options[0]
        for option in list(_widget(at, "radio", label).options):
            _widget(at, "radio", label).set_value(option).run()
            assert not _errors(at), (label, option)
        _widget(at, "radio", label).set_value(first).run()


@NEEDS_EDGES
@NEEDS_CLUSTERS
def test_cluster_page_periods_and_hubs_combine_without_errors():
    at = _go(_run(), "2")
    options = list(_widget(at, "select_slider", "Period").options)
    for period in (options[1], options[-1], options[0]):
        _widget(at, "select_slider", "Period").set_value(period).run()
        for hub in list(_widget(at, "radio", "Biggest countries").options):
            _widget(at, "radio", "Biggest countries").set_value(hub).run()
            assert not _errors(at), (period, hub)
        _widget(at, "radio", "Biggest countries").set_value(_widget(at, "radio", "Biggest countries").options[0]).run()


@NEEDS_EDGES
@NEEDS_CLUSTERS
def test_cluster_page_tabs_and_toggles_render():
    at = _go(_run(), "2")
    for label in ("Show flows between blocks", "Only countries with a certain block"):
        _widget(at, "toggle", label).set_value(True).run()
        assert not _errors(at), label
        _widget(at, "toggle", label).set_value(False).run()
    _widget(at, "radio", "Kind").set_value("Conflict").run()
    assert not _errors(at)
    _widget(at, "radio", "Show").set_value("Words only").run()
    assert not _errors(at)
