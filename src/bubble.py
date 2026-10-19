"""Words-vs-deeds bubble chart: does a country's stance toward each side of a conflict match
in what it says (verbal) and what it does (material)?

New module (does not change anything in the existing src/ files). Uses metrics.rolling_scores.
"""

import pandas as pd
import plotly.express as px

import metrics

ORANGE, BLUE = "#d35400", "#2c6fbb"


def reliability_long(conflict: str, first_id: int, second_id: int, window: int = 12,
                      min_n: int = 5) -> pd.DataFrame:
    """One row per (country, week, party): verbal/material stance toward that party, and the
    number of events behind it. `party` is the label to color by (first party's name / second
    party's name), not 'first'/'second', so the legend reads naturally."""
    first, second = conflict.split("-")
    r = metrics.rolling_scores(conflict, first_id, second_id, window=window, min_n=min_n)
    r = r[~r.actor_id.isin([first_id, second_id])]
    countries = metrics.load_countries().set_index("country_id")

    frames = []
    for party_id, party_name, suffix in [(first_id, first, "first"), (second_id, second, "second")]:
        d = r[["actor_id", "week_end", f"verbal_{suffix}", f"material_{suffix}",
               f"n_verbal_{suffix}", f"n_material_{suffix}"]].copy()
        d.columns = ["actor_id", "week_end", "verbal", "material", "n_verbal", "n_material"]
        d["party"], d["n_events"] = party_name, d.n_verbal + d.n_material
        frames.append(d)
    long = pd.concat(frames).dropna(subset=["verbal", "material"])
    long["country"] = long.actor_id.map(countries.name)
    long["week_label"] = long.week_end.dt.strftime("%Y-%m-%d")
    return long.sort_values("week_end")


def reliability_bubble(conflict: str, first_id: int, second_id: int, window: int = 12,
                        min_n: int = 5) -> "px.Figure":
    """Animated scatter: x = verbal stance, y = material stance, size = event count, color = party.
    Top-right / bottom-left = words and deeds agree; top-left / bottom-right = they contradict."""
    long = reliability_long(conflict, first_id, second_id, window, min_n)
    first, second = conflict.split("-")
    fig = px.scatter(
        long, x="verbal", y="material", size="n_events", color="party", animation_frame="week_label",
        hover_name="country", range_x=[-1.05, 1.05], range_y=[-1.05, 1.05], size_max=40,
        color_discrete_map={first: ORANGE, second: BLUE},
        labels={"verbal": "hostile ← words → friendly", "material": "hostile ← actions → friendly"},
        title=f"Words vs. deeds toward {first} and {second} (rolling {window}-week window, min {min_n} events/dot)")
    fig.add_hline(y=0, line_color="lightgray", line_width=1)
    fig.add_vline(x=0, line_color="lightgray", line_width=1)
    fig.update_layout(height=750, legend_title_text="Friendlier toward")
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    return fig
