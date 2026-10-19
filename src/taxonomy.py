"""How GENOME's 16 event types are grouped: the single source of truth for the whole project.

Follows Table 4 of the GENOME paper (words = verbal, actions = material).
Change the grouping here and every script, metric and page follows.
"""

LAYERS = {
    ("words", "cooperation"): ["AGREE", "CONSULT", "SUPPORT", "CONCEDE"],
    ("material", "cooperation"): ["AID", "COOPERATE", "RETREAT"],
    ("words", "conflict"): ["REQUEST", "ACCUSE", "REJECT", "THREATEN"],
    ("material", "conflict"): ["ASSAULT", "SANCTION", "COERCE", "MOBILIZE", "PROTEST"],
}

VERBAL_COOP = LAYERS[("words", "cooperation")]
MATERIAL_COOP = LAYERS[("material", "cooperation")]
VERBAL_CONF = LAYERS[("words", "conflict")]
MATERIAL_CONF = LAYERS[("material", "conflict")]

MATERIAL_TYPES = MATERIAL_COOP + MATERIAL_CONF
WORD_TYPES = VERBAL_COOP + VERBAL_CONF
ALL_TYPES = MATERIAL_TYPES + WORD_TYPES

_LOOKUP = {t: key for key, types in LAYERS.items() for t in types}


def layer_of(event_type: str) -> str:
    """'words' or 'material'."""
    return _LOOKUP[event_type][0]


def category_of(event_type: str) -> str:
    """'cooperation' or 'conflict'."""
    return _LOOKUP[event_type][1]
