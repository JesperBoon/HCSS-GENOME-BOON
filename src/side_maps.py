"""Conflict explorer for the notebook: one dropdown, a side-taking map and a small map of the two parties.

New module (does not change anything in the existing src/ files). Uses metrics.side_table for the scores.
"""

import pandas as pd
import plotly.graph_objects as go

import metrics

PAIRS = metrics.ROOT / "data" / "processed" / "conflict_pairs.csv"
LAYERS = {"verbal": "side_verbal", "material": "side_material"}
RED, BLUE = "#c0392b", "#2c6fbb"


def conflict_options() -> pd.DataFrame:
    """The conflicts that have dyad data, biggest hostile share first, with parties and basic numbers."""
    countries = metrics.load_countries().set_index("name")
    pairs = pd.read_csv(PAIRS)
    rows = []
    for label in metrics.conflicts_available():
        first, second = label.split("-")
        p = pairs[(pairs["first"] == first) & (pairs["second"] == second)].iloc[0]
        rows.append(dict(label=label, first=first, second=second,
                         first_id=int(countries.loc[first, "country_id"]),
                         second_id=int(countries.loc[second, "country_id"]),
                         first_iso=countries.loc[first, "iso_alpha3"],
                         second_iso=countries.loc[second, "iso_alpha3"],
                         hostile_share=p.hostile_share, conflict_events=int(p.conflict_events)))
    return pd.DataFrame(rows).sort_values("hostile_share", ascending=False).reset_index(drop=True)


def conflict_explorer(world: dict, rng: float = 2.0) -> go.Figure:
    """Dropdown 'conflict - verbal/material'. Red = leans to the first party, blue = to the second."""
    countries = metrics.load_countries().set_index("country_id")
    fig, options = go.Figure(), conflict_options()
    buttons = []
    for r in options.itertuples():
        t = metrics.side_table(r.label, r.first_id, r.second_id, metrics.quarters_available(r.label))
        t["iso"] = t.actor_id.map(countries.iso_alpha3)
        t["name"] = t.actor_id.map(countries.name)
        for layer, col in LAYERS.items():
            d = t.dropna(subset=[col])
            fig.add_trace(go.Choropleth(
                geojson=world, featureidkey="properties.ISO_A3_EH", locations=d.iso, z=d[col],
                text=d.name, colorscale="RdBu", zmin=-rng, zmax=rng, geo="geo", visible=False,
                colorbar=dict(title=f"leans to {r.first} (red)<br>/ {r.second} (blue)", len=0.7, x=0.7),
                hovertemplate="%{text}: %{z:.2f}<extra></extra>"))
        fig.add_trace(go.Choropleth(
            geojson=world, featureidkey="properties.ISO_A3_EH", locations=[r.first_iso, r.second_iso],
            z=[0, 1], colorscale=[[0, RED], [1, BLUE]], showscale=False, geo="geo2", visible=False,
            text=[r.first, r.second], hovertemplate="%{text}<extra></extra>"))
    per = 3  # traces per conflict: verbal, material, highlight
    for i, r in enumerate(options.itertuples()):
        for j, layer in enumerate(LAYERS):
            vis = [False] * len(fig.data)
            vis[i * per + j], vis[i * per + 2] = True, True
            note = (f"<b>{r.label}</b>: {r.hostile_share:.0%} of the {r.conflict_events:,} events between the two "
                    f"are conflict<br>Showing {layer} support: red leans to {r.first}, blue to {r.second}")
            buttons.append(dict(label=f"{r.label} · {layer}", method="update",
                                args=[{"visible": vis}, {"annotations": [dict(
                                    text=note, x=0, y=1.12, xref="paper", yref="paper", showarrow=False, align="left")]}]))
    for k in (0, 2):
        fig.data[k].visible = True
    geo = dict(showframe=False, showcoastlines=False, projection_type="natural earth", showland=True,
               landcolor="#e6e6e6", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        geo=dict(domain=dict(x=[0, 0.72], y=[0, 1]), **geo), geo2=dict(domain=dict(x=[0.76, 1], y=[0.25, 0.75]), **geo),
        updatemenus=[dict(buttons=buttons, x=0, y=1.0, xanchor="left", yanchor="bottom", direction="down")],
        margin=dict(l=0, r=0, t=110, b=0), height=680)
    fig.update_layout(annotations=buttons[0]["args"][1]["annotations"])
    return fig
