"""What kind of AID is it? GENOME's class AID is wide, so it is split by what the event text says.

The class holds military help, money, humanitarian relief, but also exchanges of prisoners and bodies, evacuations of
citizens, trade measures and events that only mention aid. Mixing them blurs the picture: Russia returning the bodies
of soldiers to Ukraine is coded "Russia AID Ukraine" and would make two adversaries look like cooperating partners.

Rules, checked in this order on the event summary and the core sentence (the first that matches wins):
    exchange      prisoners, hostages, detainees, bodies or remains released, returned, swapped or repatriated
    military      weapons, missiles, ammunition, air defence, drones, tanks, troops, military aid, arms, training
    (other)       ceremonies, labour, attacks on aid sites: events that only mention aid
    trade         tariffs, imports, exports, crude oil, sanctions waivers
    humanitarian  humanitarian aid, food, medicine, relief, shelter, rescue, disaster
    financial     loans, grants, funding, billions or millions, investment, budget support
    evacuation    evacuation of citizens or diplomats
    trade         oil, gas, purchases, trade (the weaker words, checked last)
    other         everything else (often not support at all)

"Support" is the union of military, humanitarian and financial. The rules are simple on purpose: they can be read, tested and
checked by hand (see tests/test_aid_types.py and the validation on a hand-labelled sample in the README).
"""

import re

CATEGORIES = ("exchange", "military", "humanitarian", "financial", "evacuation", "trade", "other")
SUPPORT = ("military", "humanitarian", "financial")

_RULES = [
    ("exchange", r"\b(prisoners?|hostages?|detainees?|POWs?|captives?|bod(?:y|ies)|remains|corpses|captured soldiers)\b.{0,80}"
                 r"\b(releas\w*|return\w*|exchang\w*|swap\w*|freed|repatriat\w*|handed over|transfer\w*)\b"
                 r"|\b(releas\w*|return\w*|exchang\w*|swap\w*|freed|repatriat\w*|handed over)\b.{0,80}"
                 r"\b(prisoners?|hostages?|detainees?|POWs?|captives?|bod(?:y|ies)|remains|corpses|captured soldiers)\b"),
    ("military", r"\b(weapons?|missiles?|ammunition|munitions?|air[- ]defen[cs]e|interceptors?|tanks?|drones?|artillery|howitzers?|"
                 r"troops?|soldiers?|fighter jets?|F-16s?|Patriot|HIMARS|military (?:aid|assistance|package|support|equipment|experts?)|"
                 r"arms|armed|rockets?|cruise missiles?|warships?|submarines?|military training|trained|intelligence sharing)\b"),
    # not support at all: events that only mention aid, labour, ceremonies, attacks on aid sites
    ("other", r"\b(commemorat\w*|statue|unveil\w*|book fair|work permits?|construction workers?|labou?r|"
              r"projectiles?.{0,80}aid distribution|aid distribution sites?.{0,80}projectiles?)\b"),
    ("trade", r"\b(tariffs?|import tariffs?|waivers?|exempt\w*|crude|barrels?|sanctions? (?:relief|waiver)|shipbuilding|"
              r"imports?|exports?)\b"),
    ("humanitarian", r"\b(humanitarian|food|medical|medicines?|medications?|relief|shelter|tents|blankets|rescue|"
                     r"earthquake|flood\w*|famine|vaccines?|emergency aid|aid (?:convoy|trucks?|shipment|flights?)|ceasefire aid|"
                     r"wheat|drinking water|water supply|electricity|winterization|hospital\w*)\b"),
    ("financial", r"\b(loans?|grants?|funding|funds?|financ\w*|invest\w*|budget|credit|billions?|millions?|donat\w*|"
                  r"assistance package|economic (?:aid|support|assistance)|reconstruction|\$\d|€\d|£\d|riyals?|dirhams?)\b"),
    ("evacuation", r"\b(evacuat\w*|repatriat\w* of (?:nationals|citizens)|nationals|citizens were (?:evacuated|flown))\b"),
    ("trade", r"\b(oil|gas|LNG|purchas\w*|trade)\b"),
]
_COMPILED = [(name, re.compile(rx, re.IGNORECASE | re.DOTALL)) for name, rx in _RULES]


def classify_aid(summary: str | None, core_sentence: str | None = None) -> str:
    """The kind of an AID event, from its summary and core sentence (see the module docstring for the rules)."""
    text = f"{summary or ''} {core_sentence or ''}"
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return "other"


def is_support(kind: str) -> bool:
    """True for military, humanitarian and financial aid."""
    return kind in SUPPORT
