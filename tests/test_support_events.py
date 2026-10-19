"""Tests for src/support_events.py."""

import pandas as pd

import support_events as se


def _events():
    return pd.DataFrame({"event_id": ["a", "b", "c", "d"], "event_type": ["AID", "AID", "COOPERATE", "ASSAULT"]})


def _text():
    return pd.DataFrame({"event_id": ["a", "b", "c", "d"],
                         "event_summary": ["The US sent a military aid package with missiles", "Russia returned the bodies of soldiers", "x", "y"],
                         "core_sentence": ["USA AID Ukraine", "Russia AID Ukraine", "A COOPERATE B", "C ASSAULT D"]})


def test_only_aid_events_get_a_kind():
    ev = se.attach_aid_kind(_events(), _text())
    assert ev.set_index("event_id").aid_kind.to_dict()["a"] == "military"
    assert ev.set_index("event_id").aid_kind.to_dict()["b"] == "exchange"
    assert ev.aid_kind.isna().tolist() == [False, False, True, True]


def test_support_aid_gets_its_own_type_and_exchanges_stay_aid():
    ev = se.mark_support(se.attach_aid_kind(_events(), _text())).set_index("event_id")
    assert ev.event_type.to_dict() == {"a": "AID_SUPPORT", "b": "AID", "c": "COOPERATE", "d": "ASSAULT"}


def test_marking_does_not_change_the_input():
    ev = se.attach_aid_kind(_events(), _text())
    se.mark_support(ev)
    assert (ev.event_type == "AID").sum() == 2


def test_types_for_the_layers():
    assert se.types_for("support", "cooperation") == ["AID_SUPPORT"]
    assert "ASSAULT" in se.types_for("support", "conflict")
    assert "AID" in se.types_for("material", "cooperation") and "AID_SUPPORT" in se.types_for("material", "cooperation")
    assert "AID_SUPPORT" not in se.types_for("words", "cooperation")
    assert "CONSULT" in se.types_for("words", "cooperation")
