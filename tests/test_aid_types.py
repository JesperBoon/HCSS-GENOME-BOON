"""Tests for src/aid_types.py: the rules on typical sentences, and agreement with hand-labelled events."""

from pathlib import Path

import pandas as pd
import pytest

import aid_types as at

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text, expected", [
    ("Russia returned the bodies of 1,212 soldiers to Ukraine", "exchange"),
    ("The two sides carried out a prisoner swap and freed 40 detainees", "exchange"),
    ("Hamas handed over three hostages in a ceremony", "exchange"),
    ("The United States announced a military aid package with missiles and ammunition", "military"),
    ("North Korea sent troops to support Russia", "military"),
    ("Iran sent humanitarian aid to Gaza with food, tents and blankets", "humanitarian"),
    ("India delivered medical supplies to a hospital in Kabul", "humanitarian"),
    ("Germany proposed a 140 billion euro loan to Ukraine", "financial"),
    ("Saudi Arabia gave 60 million dollars in budget support to Yemen", "financial"),
    ("Several countries evacuated their nationals from the region", "evacuation"),
    ("Mexico was exempted from the new tariff list", "trade"),
    ("Japan imported crude oil from Saudi Arabia", "trade"),
    ("A statue was unveiled at the embassy during a commemoration", "other"),
    ("Construction workers from India will arrive in Israel", "other"),
])
def test_typical_sentences_get_the_expected_kind(text, expected):
    assert at.classify_aid(text) == expected


def test_an_exchange_beats_military_words_in_the_same_sentence():
    assert at.classify_aid("Soldiers captured earlier were released in a prisoner exchange") == "exchange"


def test_military_beats_humanitarian_and_financial():
    assert at.classify_aid("A humanitarian and military package of 200 million with air defence systems") == "military"


def test_the_core_sentence_is_used_too():
    assert at.classify_aid("Nothing special happened", "Russia AID Ukraine prisoner release") == "exchange"
    assert at.classify_aid(None, None) == "other"


def test_support_is_military_humanitarian_or_financial_only():
    assert [at.is_support(k) for k in at.CATEGORIES] == [False, True, True, True, False, False, False]


@pytest.mark.skipif(not (ROOT / "data/processed/events_text.parquet").exists(), reason="event text not built")
@pytest.mark.parametrize("gold, minimum_exact, minimum_support", [("aid_gold.csv", 0.85, 0.85), ("aid_gold_blind.csv", 0.70, 0.80)])
def test_agreement_with_hand_labelled_events(gold, minimum_exact, minimum_support):
    """aid_gold.csv was used to improve the rules; aid_gold_blind.csv was labelled before the rules were run on it (the honest number)."""
    labels = pd.read_csv(ROOT / "tests" / gold)
    text = pd.read_parquet(ROOT / "data/processed/events_text.parquet", columns=["event_id", "event_summary", "core_sentence"])
    d = labels.merge(text, on="event_id")
    d["rule"] = [at.classify_aid(a, b) for a, b in zip(d.event_summary, d.core_sentence)]
    assert len(d) == 40
    assert (d.label == d.rule).mean() >= minimum_exact
    assert (d.label.isin(at.SUPPORT) == d.rule.isin(at.SUPPORT)).mean() >= minimum_support


def test_capital_letters_do_not_matter():
    assert at.classify_aid("RUSSIA RETURNED THE BODIES OF SOLDIERS") == "exchange"
    assert at.classify_aid("The Pentagon sent Weapons") == "military"
