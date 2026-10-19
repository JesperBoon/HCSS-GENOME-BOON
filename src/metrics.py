"""
metrics.py — reliability metrics computed from monthly_counts.csv (free count queries).

Event-type groups follow the PLOVER quad (verbal/material x cooperation/conflict):
    verbal_coop  AGREE, CONSULT, SUPPORT, CONCEDE
    mat_coop     AID, COOPERATE, RETREAT
    verbal_conf  REQUEST, ACCUSE, REJECT, THREATEN
    mat_conf     ASSAULT, MOBILIZE, COERCE, SANCTION, PROTEST   (as in the GENOME paper, Table 4)

Metrics are shares and ratios, not raw counts, because event volume roughly tripled
between 2024 and 2026 (more sources, not necessarily more world events).
"""

import json
from pathlib import Path

import pandas as pd
import requests

from taxonomy import MATERIAL_CONF, MATERIAL_COOP, VERBAL_CONF, VERBAL_COOP  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parent.parent
COUNTS = ROOT / "data" / "processed" / "monthly_counts.csv"
COUNTRIES = ROOT / "data" / "processed" / "countries.json"

GROUPS = {
    "verbal_coop": VERBAL_COOP,
    "mat_coop": MATERIAL_COOP,
    "verbal_conf": VERBAL_CONF,
    "mat_conf": MATERIAL_CONF,
}
CONFLICT_TYPES = GROUPS["verbal_conf"] + GROUPS["mat_conf"]

MIN_MONTHS_FOR_STABILITY = 4


def load_countries() -> pd.DataFrame:
    if not COUNTRIES.exists():
        COUNTRIES.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get("https://genome.app.hcss.nl/api/countries", timeout=60).json()
        COUNTRIES.write_text(json.dumps(r["countries"]))
    return pd.DataFrame(json.loads(COUNTRIES.read_text())).rename(columns={"id": "country_id"})


def load_monthly() -> pd.DataFrame:
    """Wide table: one row per country-month, one column per event type, plus groups."""
    df = pd.read_csv(COUNTS)
    wide = df.pivot_table(index=["country_id", "month"], columns="event_type",
                          values="n", aggfunc="sum").fillna(0).reset_index()
    for g, types in GROUPS.items():
        wide[g] = wide[types].sum(axis=1)
    wide["conflict"] = wide[CONFLICT_TYPES].sum(axis=1)
    wide["total"] = wide[[c for c in df.event_type.unique()]].sum(axis=1)
    return wide


def window(monthly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return monthly[(monthly.month >= start) & (monthly.month <= end)]


def country_profile(monthly: pd.DataFrame, start: str, end: str, min_events: int = 30) -> pd.DataFrame:
    """One row per country for the chosen window.

    conflict_share   share of the country's events (as actor) that are conflict
    acts_per_word_conf  material conflict events per verbal conflict event
    acts_per_word_coop  material cooperation events per verbal cooperation event
    stability_cv     std/mean of the monthly conflict share (lower = steadier posture)
    """
    w = window(monthly, start, end)
    g = w.groupby("country_id")
    out = g[["total", "conflict", "verbal_coop", "mat_coop", "verbal_conf", "mat_conf"]].sum()
    out["conflict_share"] = out.conflict / out.total
    out["acts_per_word_conf"] = out.mat_conf / out.verbal_conf.where(out.verbal_conf > 0)
    out["acts_per_word_coop"] = out.mat_coop / out.verbal_coop.where(out.verbal_coop > 0)

    w = w[w.total > 0].assign(share=lambda d: d.conflict / d.total)
    stats = w.groupby("country_id").share.agg(["mean", "std", "count"])
    cv = (stats["std"] / stats["mean"]).where(stats["count"] >= MIN_MONTHS_FOR_STABILITY)
    out["stability_cv"] = cv
    out["months"] = stats["count"]
    return out[out.total >= min_events].reset_index()


def monthly_series(monthly: pd.DataFrame, country_id: int) -> pd.DataFrame:
    d = monthly[monthly.country_id == country_id].sort_values("month").copy()
    d["conflict_share"] = d.conflict / d.total.where(d.total > 0)
    return d


# ── Dyad stance and side-taking (data from scripts/build_dyad_counts.py) ─────────────

DYADS = ROOT / "data" / "processed" / "dyad_counts.csv"


def _stance(pos: pd.Series, neg: pd.Series, min_n: int) -> pd.Series:
    """(pos - neg) / (pos + neg): +1 all friendly, -1 all hostile; NaN if too few events."""
    tot = pos + neg
    return ((pos - neg) / tot).where(tot >= min_n)


def conflicts_available() -> list[str]:
    """Conflict labels ('First-Second') present in dyad_counts.csv."""
    return sorted(pd.read_csv(DYADS, usecols=["conflict"]).conflict.unique()) if DYADS.exists() else []


def quarters_available(conflict: str) -> list[str]:
    d = pd.read_csv(DYADS, usecols=["conflict", "quarter"])
    return sorted(d[d.conflict == conflict].quarter.unique())


# Severity weights per event type: the midpoint of the intensity band the GENOME methodology
# (Table 3) assigns to the type. Intensity is "driven mainly by the event type", and counts
# cannot give per-event intensity, so a type-level weight is the best free approximation.
TYPE_WEIGHTS = {
    # routine / verbal band, 0.0-0.3
    "CONSULT": 0.15, "REQUEST": 0.15, "ACCUSE": 0.15, "REJECT": 0.15,
    # formal / non-lethal band, 0.4-0.7
    "AGREE": 0.55, "SUPPORT": 0.55, "CONCEDE": 0.55,
    "THREATEN": 0.55, "SANCTION": 0.55, "PROTEST": 0.55, "MOBILIZE": 0.55, "COERCE": 0.55,
    # material / lethal band, 0.8-1.0
    "AID": 0.9, "COOPERATE": 0.9, "RETREAT": 0.9, "ASSAULT": 0.9,
}


def side_table(conflict: str, first_id: int, second_id: int, quarters: list[str],
               min_n: int = 3, weighted: bool = False) -> pd.DataFrame:
    """Where does every country stand in the conflict First vs Second?

    For each actor A and each target T in {first, second}, over the chosen quarters:
        all      all cooperation vs all conflict events A -> T
        verbal   verbal cooperation vs verbal conflict (words)
        material material cooperation vs material conflict (acts)
    stance = (cooperation - conflict) / (cooperation + conflict), in [-1, 1].
    With weighted=True every event counts with the severity weight of its type
    (TYPE_WEIGHTS) instead of 1, so one airstrike outweighs a routine meeting.
    side_* = stance toward SECOND minus stance toward FIRST, in [-2, 2]:
        > 0 leans to the second party, < 0 leans to the first, ~0 no side.
    A score exists only if A has >= min_n (unweighted) events of that kind toward BOTH parties.
    gap = side_material - side_verbal: what a country does minus what it says.
    """
    d = pd.read_csv(DYADS)
    d = d[(d.conflict == conflict) & d.quarter.isin(quarters)]
    w = d.groupby(["actor_id", "target_id", "key"]).n.sum().unstack("key").fillna(0)
    missing = [t for t in TYPE_WEIGHTS if t not in w.columns]
    if missing:
        raise RuntimeError(f"dyad_counts.csv lacks event types {missing}; "
                           f"rerun scripts/build_dyad_counts.py for {conflict}.")

    def tot(types, weight):
        return sum(w[t] * (TYPE_WEIGHTS[t] if weight else 1.0) for t in types)

    groups = {"vc": VERBAL_COOP, "mc": MATERIAL_COOP, "vf": VERBAL_CONF, "mf": MATERIAL_CONF}
    raw = {g: tot(ts, False) for g, ts in groups.items()}
    val = {g: tot(ts, weighted) for g, ts in groups.items()}
    kinds = {"all": (["vc", "mc"], ["vf", "mf"]), "verbal": (["vc"], ["vf"]), "material": (["mc"], ["mf"])}

    out = pd.DataFrame(index=w.index.get_level_values("actor_id").unique())
    for kind, (pos_g, neg_g) in kinds.items():
        pos_r, neg_r = sum(raw[g] for g in pos_g), sum(raw[g] for g in neg_g)
        pos_v, neg_v = sum(val[g] for g in pos_g), sum(val[g] for g in neg_g)
        tot_v = pos_v + neg_v
        stance = (((pos_v - neg_v) / tot_v.where(tot_v > 0)).where(pos_r + neg_r >= min_n)).unstack("target_id")
        n = (pos_r + neg_r).unstack("target_id")
        out[f"n_{kind}_first"], out[f"n_{kind}_second"] = n[first_id], n[second_id]
        out[f"{kind}_first"], out[f"{kind}_second"] = stance[first_id], stance[second_id]
        out[f"side_{kind}"] = stance[second_id] - stance[first_id]
    out["gap"] = out.side_material - out.side_verbal
    # Acts are more extreme than words by construction (aid vs. sanctions), so most countries
    # show a positive gap. Rank by the deviation from the typical country instead.
    out["gap_vs_typical"] = out.gap - out.gap.median()
    return out.reset_index()


PATTERNS = {
    "++": "Friendly words, friendly actions",
    "--": "Hostile words, hostile actions",
    "+-": "Friendly words, hostile actions",
    "-+": "Hostile words, friendly actions",
    "mixed": "Mixed or neutral",
}


def pattern(words: float, acts: float, dead: float = 0.2) -> str:
    """Do words and actions toward the same country point the same way?

    Stances within +-dead of zero count as neutral, so a pattern is only named when both
    words and actions are clearly friendly or clearly hostile.
    """
    sw = 1 if words >= dead else -1 if words <= -dead else 0
    sa = 1 if acts >= dead else -1 if acts <= -dead else 0
    if sw == 0 or sa == 0:
        return PATTERNS["mixed"]
    return PATTERNS[("+" if sw > 0 else "-") + ("+" if sa > 0 else "-")]


# ── Weekly data and rolling windows (scripts/build_weekly_counts.py) ─────────────────

WEEKLY_DIR = ROOT / "data" / "processed" / "weekly"


def weekly_available() -> list[str]:
    """Conflict labels for which weekly data has been downloaded completely enough to use."""
    return sorted(p.stem for p in WEEKLY_DIR.glob("*.csv")) if WEEKLY_DIR.exists() else []


def load_weekly(conflict: str) -> pd.DataFrame:
    d = pd.read_csv(WEEKLY_DIR / f"{conflict}.csv", parse_dates=["week_start"])
    return d[d.key != "_done"]


def weekly_players(conflict: str) -> list[int]:
    """Players in the weekly file, biggest first (most events toward the two parties)."""
    d = load_weekly(conflict)
    return d.groupby("actor_id").n.sum().sort_values(ascending=False).index.tolist()


def rolling_scores(conflict: str, first_id: int, second_id: int, window: int = 8,
                   min_n: int = 3, weighted: bool = False) -> pd.DataFrame:
    """Side scores over time with a rolling window of `window` weeks, moving one week at a time.

    One row per (actor, week). `week_end` is the Sunday of the last week in the window.
    Columns per kind in {all, verbal, material}:
        n_<kind>_first / _second   real (unweighted) events toward each party inside the window
        <kind>_first / _second     stance toward each party, NaN if fewer than min_n events
        side_<kind>                stance toward second minus stance toward first
    Weeks before a full window exists are NaN, and so is any score without enough events.
    """
    d = load_weekly(conflict)
    weeks = pd.date_range(d.week_start.min(), d.week_start.max(), freq="7D")
    weights = pd.Series(TYPE_WEIGHTS)
    groups = {"vc": VERBAL_COOP, "mc": MATERIAL_COOP, "vf": VERBAL_CONF, "mf": MATERIAL_CONF}
    kinds = {"all": (["vc", "mc"], ["vf", "mf"]), "verbal": (["vc"], ["vf"]), "material": (["mc"], ["mf"])}

    frames = []
    for (actor, target), g in d.groupby(["actor_id", "target_id"]):
        w = (g.pivot_table(index="week_start", columns="key", values="n", aggfunc="sum")
             .reindex(index=weeks, columns=list(TYPE_WEIGHTS)).fillna(0))
        raw = {k: w[ts].sum(axis=1) for k, ts in groups.items()}
        val = {k: (w[ts] * weights[ts]).sum(axis=1) if weighted else raw[k] for k, ts in groups.items()}
        res = pd.DataFrame(index=weeks)
        for kind, (pos_g, neg_g) in kinds.items():
            pos_r = sum(raw[k] for k in pos_g).rolling(window).sum()
            neg_r = sum(raw[k] for k in neg_g).rolling(window).sum()
            pos_v = sum(val[k] for k in pos_g).rolling(window).sum()
            neg_v = sum(val[k] for k in neg_g).rolling(window).sum()
            tot_v = pos_v + neg_v
            res[f"n_{kind}"] = pos_r + neg_r
            res[kind] = ((pos_v - neg_v) / tot_v.where(tot_v > 0)).where(pos_r + neg_r >= min_n)
        res["actor_id"], res["target_id"] = actor, target
        frames.append(res.rename_axis("week_start").reset_index())
    long = pd.concat(frames)

    out = long[["actor_id", "week_start"]].drop_duplicates().set_index(["actor_id", "week_start"])
    for kind in kinds:
        stance = long.pivot_table(index=["actor_id", "week_start"], columns="target_id", values=kind, dropna=False)
        n = long.pivot_table(index=["actor_id", "week_start"], columns="target_id", values=f"n_{kind}", dropna=False)
        out[f"n_{kind}_first"], out[f"n_{kind}_second"] = n.get(first_id), n.get(second_id)
        out[f"{kind}_first"], out[f"{kind}_second"] = stance.get(first_id), stance.get(second_id)
        out[f"side_{kind}"] = stance.get(second_id) - stance.get(first_id)
    out = out.reset_index()
    out["week_end"] = out.week_start + pd.Timedelta(days=6)
    return out.sort_values(["actor_id", "week_start"]).reset_index(drop=True)
