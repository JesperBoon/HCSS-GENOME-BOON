"""Page 3: a rebuild of the kind of world map GENOME itself offers, by month."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import metrics as mt
from views.common import Data, recent_events, world_geojson, your_words


def render(data: Data) -> None:
    countries, monthly, months = data.countries, data.monthly, data.months
    st.title("Step 3 · Rebuilding the GENOME map")

    your_words("map_why")
    st.divider()

    left, right = st.columns([3, 2], gap="large")

    with left:
        metric_opts = {
            "Events (as actor)": ("total", "Blues"),
            "Conflict share": ("conflict_share", "Reds"),
        }
        metric_label = st.selectbox("Colour the map by", list(metric_opts))
        col, scale = metric_opts[metric_label]
        start, end = st.select_slider("Time window (month)", options=months,
                                      value=(months[0], months[-1]))
        prof = mt.country_profile(monthly, start, end).merge(
            countries[["country_id", "name", "iso_alpha3"]], on="country_id")

        vals = prof[col]
        fig = px.choropleth_map(
            prof, geojson=world_geojson(), featureidkey="properties.ISO_A3_EH",
            locations="iso_alpha3", color=col, hover_name="name",
            hover_data={"iso_alpha3": False, "total": ":,", "conflict_share": ":.0%"},
            color_continuous_scale=scale, map_style="carto-positron",
            zoom=0.9, center={"lat": 25, "lon": 10}, opacity=0.8,
            range_color=(vals.min(), vals.max()),
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=480,
                          coloraxis_colorbar=dict(title="", thickness=12, len=0.6))
        sel = st.plotly_chart(fig, width="stretch", on_select="rerun",
                              selection_mode="points", key="worldmap")
        st.caption(f"{len(prof)} countries with ≥30 events as actor in {start} → {end}. "
                   "Click a country, or pick one on the right.")

        # A click on the map sets the country, but only once per new click.
        pts = sel.selection.points if sel and sel.selection else []
        if pts:
            iso = pts[0].get("location")
            if iso and iso != st.session_state.get("last_click"):
                st.session_state["last_click"] = iso
                hit = countries[countries.iso_alpha3 == iso]
                if not hit.empty:
                    st.session_state["country_name"] = hit.iloc[0]["name"]

    with right:
        options = sorted(prof["name"])
        st.session_state.setdefault("country_name", "Iran" if "Iran" in options else options[0])
        if st.session_state["country_name"] not in options:
            st.session_state["country_name"] = options[0]
        chosen = st.selectbox("Country", options, key="country_name")
        row = prof[prof.name == chosen].iloc[0]
        cid = int(row.country_id)

        st.subheader(chosen)
        k1, k2 = st.columns(2)
        k1.metric("Events as actor", f"{int(row.total):,}")
        k2.metric("Conflict share", f"{row.conflict_share:.0%}")

        ser = mt.monthly_series(monthly, cid)
        ser = ser[(ser.month >= start) & (ser.month <= end)]
        bars = go.Figure()
        for g, colr, lbl in [("verbal_coop", "#A6BDDB", "Cooperation: words"),
                             ("mat_coop", "#003082", "Cooperation: acts"),
                             ("verbal_conf", "#F2B8A2", "Conflict: words"),
                             ("mat_conf", "#C0392B", "Conflict: acts")]:
            bars.add_bar(x=ser.month, y=ser[g], name=lbl, marker_color=colr)
        bars.update_layout(barmode="stack", height=230, margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                           yaxis_title="Events / month")
        st.plotly_chart(bars, width="stretch")

    # events the country is involved in
    st.subheader(f"Newest events involving {chosen}")
    evs = recent_events(cid)[:40]
    if evs:
        table = pd.DataFrame([{
            "Date": e["event_date"], "Event": e["core_sentence"], "Type": e["event_type"],
            "Intensity": e["intensity"], "Reports": e["article_count"],
        } for e in evs])
        st.dataframe(table, hide_index=True, width="stretch", height=300)
        with st.expander("Read the summary of the newest event"):
            st.write(evs[0]["event_summary"])
    st.caption("Free page-1 query (newest first). Older events need API credits.")
    your_words("map_reading")
