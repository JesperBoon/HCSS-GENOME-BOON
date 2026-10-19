"""Events with the kind of their AID attached, and the 'support only' version of the action layer.

The paper's material-cooperation layer is AID + COOPERATE + RETREAT. Here AID events that are not support (exchanges of
prisoners and bodies, evacuations, trade measures, ceremonies; see aid_types.py) can be told apart, and the 'support only'
layer keeps only military, humanitarian and financial AID. Such events get the pseudo event type AID_SUPPORT.
"""

from pathlib import Path

import pandas as pd

import aid_types as at

SUPPORT_TYPE = "AID_SUPPORT"
ROOT = Path(__file__).resolve().parent.parent


def attach_aid_kind(events: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """Add the column aid_kind (NaN for events that are not AID). `text` has event_id, event_summary, core_sentence."""
    aid = events[events.event_type == "AID"].merge(text[["event_id", "event_summary", "core_sentence"]], on="event_id")
    kind = pd.Series([at.classify_aid(s, c) for s, c in zip(aid.event_summary, aid.core_sentence)], index=aid.event_id)
    return events.assign(aid_kind=events.event_id.map(kind))


def mark_support(events: pd.DataFrame) -> pd.DataFrame:
    """AID events whose kind is military, humanitarian or financial get the type AID_SUPPORT; other AID events stay AID."""
    out = events.copy()
    out.loc[out.event_type.eq("AID") & out.aid_kind.isin(at.SUPPORT), "event_type"] = SUPPORT_TYPE
    return out


def load_events(root: Path = ROOT) -> pd.DataFrame:
    """The stored events with aid_kind and AID_SUPPORT marked."""
    ev = pd.read_csv(root / "data" / "processed" / "edges" / "events.csv")
    text = pd.read_parquet(root / "data" / "processed" / "events_text.parquet", columns=["event_id", "event_summary", "core_sentence"])
    return mark_support(attach_aid_kind(ev, text))


def types_for(layer: str, category: str) -> list:
    """Event types of one layer. 'support' = AID_SUPPORT only; 'material' = the paper's layer; 'words' = the verbal layer."""
    import taxonomy as tx
    if layer == "support":
        return [SUPPORT_TYPE] if category == "cooperation" else tx.MATERIAL_CONF
    base = [t for t in tx.ALL_TYPES if tx.layer_of(t) == layer and tx.category_of(t) == category]
    return base + [SUPPORT_TYPE] if "AID" in base else base          # marked support AID still belongs to the paper's layer
