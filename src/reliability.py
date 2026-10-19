"""Reliability score: does a country's words now predict its actions later, toward one target?

New module (does not change anything in the existing src/ files). Built on metrics.side_table,
reused one quarter at a time, for the four known conflicts in dyad_counts.csv.

For an actor A and a target T (one of the two parties to a conflict), and three periods A, B, C
(by default the last three complete quarters):
    verbal_P, material_P    A's verbal / material stance toward T in period P, in [-1, 1]
                            ((cooperation - conflict) / (cooperation + conflict))
Six differences (verbal minus material, same period and across periods):
    d1 = verbal_A - material_A   d2 = verbal_B - material_B   d3 = verbal_C - material_C
    d4 = verbal_A - material_B   d5 = verbal_B - material_C   d6 = verbal_A - material_C
    total_reliability       mean(d1..d6): 0 = words and deeds track each other overall
    commitment_reliability  mean(d4, d5, d6): do earlier words predict later actions?
A row exists only where the actor has a verbal AND a material score (min_n events each,
default 3) in EVERY one of the three periods toward that target - this is strict by design,
so the countries that do get a score are the ones with a steady, sizeable relationship.
"""

import pandas as pd
import plotly.express as px

import metrics

CONFLICTS = {
    "Russia-Ukraine": (143, 181), "Iran-Israel": (80, 83),
    "Israel-Palestine": (83, 195), "China-USA": (36, 185),
}
DEFAULT_PERIODS = ("2026Q1", "2026Q2", "2026Q3")  # last three complete quarters


def dyad_reliability(conflict: str, periods: tuple[str, str, str] = DEFAULT_PERIODS,
                     min_n: int = 3) -> pd.DataFrame:
    """One row per (actor, target) with a reliability score toward that one target."""
    first_id, second_id = CONFLICTS[conflict]
    letters = ["A", "B", "C"]
    tables = {letter: metrics.side_table(conflict, first_id, second_id, [q], min_n=min_n)
             for letter, q in zip(letters, periods)}

    rows = []
    for target_id, target_name, suffix in [(first_id, conflict.split("-")[0], "first"),
                                            (second_id, conflict.split("-")[1], "second")]:
        merged = None
        for letter in letters:
            t = tables[letter][["actor_id", f"verbal_{suffix}", f"material_{suffix}"]]
            t.columns = ["actor_id", f"verbal_{letter}", f"material_{letter}"]
            merged = t if merged is None else merged.merge(t, on="actor_id")
        merged = merged.dropna()
        merged["conflict"], merged["target"] = conflict, target_name
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)

    out["d1"] = out.verbal_A - out.material_A
    out["d2"] = out.verbal_B - out.material_B
    out["d3"] = out.verbal_C - out.material_C
    out["d4"] = out.verbal_A - out.material_B
    out["d5"] = out.verbal_B - out.material_C
    out["d6"] = out.verbal_A - out.material_C
    out["total_reliability"] = out[["d1", "d2", "d3", "d4", "d5", "d6"]].mean(axis=1)
    out["commitment_reliability"] = out[["d4", "d5", "d6"]].mean(axis=1)
    return out


def all_conflicts_reliability(periods: tuple[str, str, str] = DEFAULT_PERIODS,
                              min_n: int = 3) -> pd.DataFrame:
    """dyad_reliability for all four known conflicts, concatenated."""
    return pd.concat([dyad_reliability(c, periods, min_n) for c in CONFLICTS], ignore_index=True)


def actor_reliability(periods: tuple[str, str, str] = DEFAULT_PERIODS, min_n: int = 3) -> pd.DataFrame:
    """One row per country: its reliability score averaged over every target it has a score
    toward (across the four conflicts) - at most 7 targets (Russia, Ukraine, Iran, Israel,
    Palestine, China, USA), at least 1. `n_targets` says how many went into the average."""
    long = all_conflicts_reliability(periods, min_n)
    countries = metrics.load_countries().set_index("country_id")
    out = long.groupby("actor_id").agg(
        total_reliability=("total_reliability", "mean"),
        commitment_reliability=("commitment_reliability", "mean"),
        n_targets=("target", "nunique"),
    ).reset_index()
    out["name"] = out.actor_id.map(countries.name)
    out["iso_alpha3"] = out.actor_id.map(countries.iso_alpha3)
    return out


def reliability_map(world: dict, periods: tuple[str, str, str] = DEFAULT_PERIODS,
                    min_n: int = 3) -> "px.Figure":
    """World map of total_reliability. Scoped to the countries with a steady enough relationship
    (every period, both layers, min_n events) toward one of the four conflicts' two parties - so
    far fewer than 242 countries. Positive (blue) = friendlier in words than in actions;
    negative (red) = friendlier in actions than in words."""
    a = actor_reliability(periods, min_n)
    fig = px.choropleth(
        a, geojson=world, locations="iso_alpha3", featureidkey="properties.ISO_A3_EH",
        color="total_reliability", color_continuous_scale="RdBu", range_color=(-1, 1),
        hover_name="name", hover_data={"n_targets": True, "iso_alpha3": False},
        labels={"total_reliability": "words friendlier (+) /<br>actions friendlier (-)"},
        title=f"Reliability score: words vs. actions, {periods[0]}-{periods[-1]} "
              f"({len(a)} countries with enough steady data)")
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=600)
    return fig
