"""Page 1: starting analysis.

A  Conflict landscape      B  Movement over time
C  Where everyone stands   D  Words versus actions

Each section carries its own inline conflict selector — no global setting changes the full page.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import geo as geo_module
import metrics as mt
from views.common import (NO_SIDE, PRIMARY, Data, load_pairs, rolling_scores,
                          side_chart, side_scores, world_geojson, your_words)

FIRST_COLOR   = "#C62828"   # leans toward the first party  (deep red)
SECOND_COLOR  = PRIMARY     # leans toward the second party (HCSS blue)
NEUTRAL_COLOR = "#90A4AE"   # no clear side (slate)

WINDOW = 12   # rolling-window size fixed at 12 weeks — smooth, consistent, decided for the viewer


# ─────────────────────────────────────────────────────────────────────────────

def render(data: Data) -> None:
    st.title("Step 1 · Starting analysis")
    your_words("intro")
    _four_categories()
    _conflict_landscape(data)
    _movement(data)
    _standing(data)
    _verbal_vs_material(data)


# ─────────────────────────────────────────────────────────────────────────────
# Opening: the four-category framework
# ─────────────────────────────────────────────────────────────────────────────

def _four_categories() -> None:
    st.markdown(
        "GENOME converts news articles into structured events — *who did what to whom*, on a date. "
        "Every event is classified on **two axes**: **verbal vs material** and "
        "**cooperation vs conflict**. This gives four categories:"
    )

    _c = "border-radius:10px;padding:18px 20px;margin-bottom:6px;line-height:1.55;font-family:sans-serif"

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown(
            f"<div style='background:#EBF2FF;{_c}'>"
            "<div style='font-size:0.72em;letter-spacing:.07em;color:#003082;"
            "font-weight:700;text-transform:uppercase;margin-bottom:6px'>Verbal · Cooperation</div>"
            "<div style='font-size:0.95em;font-weight:600;color:#003082;margin-bottom:4px'>"
            "AGREE · CONSULT · SUPPORT · CONCEDE</div>"
            "<div style='font-size:0.82em;color:#444'>Statements, agreements, diplomatic meetings, "
            "expressions of support</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='background:#EBF2FF;{_c}'>"
            "<div style='font-size:0.72em;letter-spacing:.07em;color:#003082;"
            "font-weight:700;text-transform:uppercase;margin-bottom:6px'>Material · Cooperation</div>"
            "<div style='font-size:0.95em;font-weight:600;color:#003082;margin-bottom:4px'>"
            "AID · COOPERATE · RETREAT</div>"
            "<div style='font-size:0.82em;color:#444'>Actual aid transfers, joint operations, "
            "military withdrawals</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns(2, gap="small")
    with col3:
        st.markdown(
            f"<div style='background:#FDECEA;{_c}'>"
            "<div style='font-size:0.72em;letter-spacing:.07em;color:#C62828;"
            "font-weight:700;text-transform:uppercase;margin-bottom:6px'>Verbal · Conflict</div>"
            "<div style='font-size:0.95em;font-weight:600;color:#C62828;margin-bottom:4px'>"
            "REQUEST · ACCUSE · REJECT · THREATEN</div>"
            "<div style='font-size:0.82em;color:#444'>Accusations, threats, public demands, "
            "rejections</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div style='background:#FDECEA;{_c}'>"
            "<div style='font-size:0.72em;letter-spacing:.07em;color:#C62828;"
            "font-weight:700;text-transform:uppercase;margin-bottom:6px'>Material · Conflict</div>"
            "<div style='font-size:0.95em;font-weight:600;color:#C62828;margin-bottom:4px'>"
            "ASSAULT · SANCTION · COERCE · MOBILIZE · PROTEST</div>"
            "<div style='font-size:0.82em;color:#444'>Military action, economic sanctions, "
            "coercion, troop mobilisation</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "Separating **words** from **actions** is the key move. "
        "A country can express friendly intentions while imposing sanctions — or stay silent while sending aid. "
        "This page explores what that separation reveals about how countries actually behave in conflicts."
    )


# ─────────────────────────────────────────────────────────────────────────────
# A · Conflict landscape
# ─────────────────────────────────────────────────────────────────────────────

def _conflict_landscape(data: Data) -> None:
    st.divider()
    st.subheader("A · Conflict landscape")
    st.markdown(
        "The country pairs with the most mutual conflict in GENOME since January 2024, ranked by "
        "**hostile share** — how large a fraction of all events between the pair is conflict rather "
        "than cooperation. Click a row to see it on the map."
    )

    pairs = load_pairs()
    have = set(mt.conflicts_available())

    pairs = pairs.sort_values("hostile_share", ascending=False).reset_index(drop=True)
    pairs["Hostile %"] = (pairs["hostile_share"] * 100).round(1)
    pairs["Has full data"] = (pairs["first"] + "-" + pairs["second"]).isin(have) | \
                             (pairs["second"] + "-" + pairs["first"]).isin(have)

    show = pairs[["first", "second", "Hostile %", "conflict_events"]].rename(columns={
        "first": "Country A", "second": "Country B",
        "conflict_events": "Conflict events",
    })

    left, right = st.columns([2, 3], gap="large")

    with left:
        event = st.dataframe(
            show, hide_index=True, on_select="rerun", selection_mode="single-row",
            column_config={
                "Hostile %": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100,
                    help="Conflict events as % of all events between the two countries.",
                ),
                "Conflict events": st.column_config.NumberColumn(
                    format="%d",
                    help="Conflict events in both directions since Jan 2024.",
                ),
            },
            height=420,
        )
        sel_rows = event.selection.rows if (event and event.selection) else []
        idx = sel_rows[0] if sel_rows else 0

    selected = pairs.iloc[idx]
    first_name  = selected["first"]
    second_name = selected["second"]

    with right:
        _mini_map(data, first_name, second_name)
        m1, m2 = st.columns(2)
        m1.metric("Hostile share", f"{selected['Hostile %']:.0f}%")
        m2.metric("Conflict events", f"{int(selected['conflict_events']):,}")
        st.caption(
            f"**{first_name}** (red) · **{second_name}** (blue) · "
            "'Conflict' includes verbal events such as accusations, not only violence."
        )

    st.caption(
        f"Top {len(pairs)} pairs by hostile share. "
        "Ordered so that mutual conflict pairs come first. "
        "Source: `scripts/find_conflict_pairs.py` (free count queries)."
    )


@st.cache_data(show_spinner=False)
def _get_centroids(_geojson_hash: int) -> dict:
    """ISO-3 → (lat, lon). Keyed by a hash of the geojson so Streamlit can cache it."""
    from views.common import world_geojson as _wgj
    return geo_module.country_centroids(_wgj())


def _mini_map(data: Data, first_name: str, second_name: str) -> None:
    countries_df = data.countries
    centroids = geo_module.country_centroids(world_geojson())

    first_row  = countries_df[countries_df.name == first_name]
    second_row = countries_df[countries_df.name == second_name]

    fig = go.Figure()

    if not first_row.empty and not second_row.empty:
        f_iso = first_row.iloc[0].iso_alpha3
        s_iso = second_row.iloc[0].iso_alpha3

        fig.add_trace(go.Choropleth(
            locations=[f_iso], z=[1],
            colorscale=[[0, FIRST_COLOR], [1, FIRST_COLOR]],
            showscale=False, locationmode="ISO-3",
            marker_line_width=0.8, marker_line_color="#fff",
        ))
        fig.add_trace(go.Choropleth(
            locations=[s_iso], z=[1],
            colorscale=[[0, SECOND_COLOR], [1, SECOND_COLOR]],
            showscale=False, locationmode="ISO-3",
            marker_line_width=0.8, marker_line_color="#fff",
        ))

        f_coords = centroids.get(f_iso)
        s_coords = centroids.get(s_iso)
        if f_coords and s_coords:
            lats, lons = geo_module.arc(f_coords[0], f_coords[1], s_coords[0], s_coords[1])
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons, mode="lines",
                line=dict(color="#555", width=1.8, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
            for nm, lat, lon, col in [
                (first_name,  f_coords[0], f_coords[1], FIRST_COLOR),
                (second_name, s_coords[0], s_coords[1], SECOND_COLOR),
            ]:
                fig.add_trace(go.Scattergeo(
                    lat=[lat], lon=[lon], mode="markers+text",
                    text=[nm], textposition="top center",
                    textfont=dict(size=11, color="#1A1A1A"),
                    marker=dict(size=10, color=col, line=dict(color="white", width=1.5)),
                    showlegend=False, hoverinfo="skip",
                ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#E8ECF2",
        showocean=True, oceancolor="#D4E4F0",
        showframe=False, showcountries=True, countrycolor="#C8CDD8",
        showcoastlines=False, bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=290, showlegend=False)
    st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# B · Movement over time
# ─────────────────────────────────────────────────────────────────────────────

def _movement(data: Data) -> None:
    st.divider()
    st.subheader("B · Movement over time")
    st.markdown(
        "A single score over the whole period hides countries that change course. "
        "A **12-week rolling window** slides over the data, one week at a time — "
        "revealing how each country's position shifts between the two conflict parties over time."
    )
    your_words("movement")

    have = mt.conflicts_available()
    if not have:
        st.info("No conflict data available. Run `scripts/build_dyad_counts.py` first.")
        return

    cname = st.selectbox(
        "Conflict", have,
        index=have.index("Russia-Ukraine") if "Russia-Ukraine" in have else 0,
        key="b_conflict",
    )
    first_name, second_name = cname.split("-", 1)
    ids = dict(zip(data.countries.name, data.countries.country_id))
    first_id, second_id = ids[first_name], ids[second_name]

    if cname not in mt.weekly_available():
        st.info(
            f"Weekly data for **{cname}** is not downloaded yet. "
            f"Run: `.venv/bin/python scripts/build_weekly_counts.py {first_name} {second_name}` (free)."
        )
        return

    rs = rolling_scores(
        cname, first_id, second_id, WINDOW, 3, False,
        (mt.WEEKLY_DIR / f"{cname}.csv").stat().st_mtime,
    )
    rs = rs[rs.week_start >= rs.week_start.min() + pd.Timedelta(weeks=WINDOW - 1)].copy()
    rs["name"] = rs.actor_id.map(data.name_of)
    order = [data.name_of[a] for a in mt.weekly_players(cname)]
    covered = rs.groupby("name")["side_all"].apply(lambda s: s.notna().mean())

    st.caption(
        f"**{len(order)} players** · {WINDOW}-week rolling window · "
        f"last window ends {rs.week_end.max().date()} · "
        f"average coverage {covered.mean():.0%} of weeks"
    )

    tab_lines, tab_anim = st.tabs(["Lines per country", "Animation: position over time"])

    # ── lines ────────────────────────────────────────────────────────────────
    with tab_lines:
        picked = st.multiselect(
            "Countries", order, default=order[:6],
            placeholder="Type to search…", key="b_lines_pick",
        )
        fig = go.Figure()
        fig.add_hrect(y0=-NO_SIDE, y1=NO_SIDE, fillcolor="rgba(144,164,174,0.15)", line_width=0)
        for nm_ in picked:
            dd = rs[rs.name == nm_]
            fig.add_trace(go.Scatter(
                x=dd.week_end, y=dd["side_all"], mode="lines", name=nm_,
                connectgaps=False, line=dict(width=2.5),
                hovertemplate=f"<b>{nm_}</b><br>%{{x|%d %b %Y}}<br>score %{{y:.2f}}<extra></extra>",
            ))
        fig.add_hline(y=0, line_width=1, line_color="#1A1A1A")
        fig.update_layout(
            height=460, margin=dict(l=0, r=10, t=10, b=0), hovermode="closest",
            yaxis=dict(title=f"← {first_name}   side score   {second_name} →",
                       range=[-2.1, 2.1], gridcolor="#E8ECF2"),
            xaxis=dict(title=f"End of the {WINDOW}-week window", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Grey band = no clear side (|score| < {NO_SIDE}). "
            "Gaps = too few events toward both parties in that window."
        )

    # ── animation: position on axis ──────────────────────────────────────────
    with tab_anim:
        st.markdown(
            "Press **▶ Play**, or drag the slider. Each dot is a country; "
            "faint dots show the previous four positions."
        )
        weeks_ = sorted(rs.week_end.unique())
        piv = rs.pivot(index="week_end", columns="name", values="side_all").reindex(columns=order)

        def dot_colours(vals):
            return [FIRST_COLOR if v < -NO_SIDE
                    else SECOND_COLOR if v > NO_SIDE
                    else NEUTRAL_COLOR for v in vals]

        frames = []
        for i, w in enumerate(weeks_):
            cur = piv.loc[w]
            tx, ty = [], []
            for k in range(1, 5):
                if i - k >= 0:
                    prev = piv.loc[weeks_[i - k]]
                    tx += prev.tolist()
                    ty += order
            frames.append(go.Frame(name=str(pd.Timestamp(w).date()), traces=[0, 1], data=[
                go.Scatter(
                    x=tx, y=ty, mode="markers",
                    marker=dict(size=7, color=f"rgba(144,164,174,0.28)"),
                ),
                go.Scatter(
                    x=cur.tolist(), y=order, mode="markers",
                    marker=dict(size=14, color=dot_colours(cur.fillna(0)),
                                line=dict(color="white", width=1.5)),
                    hovertemplate="<b>%{y}</b><br>score %{x:.2f}<extra></extra>",
                ),
            ]))

        first_frame = frames[0].data
        fig = go.Figure(data=[first_frame[0], first_frame[1]], frames=frames)
        step_labels = [str(pd.Timestamp(w).date()) if i % 13 == 0 else ""
                       for i, w in enumerate(weeks_)]
        fig.update_layout(
            height=max(520, 20 * len(order) + 160),
            margin=dict(l=0, r=10, t=10, b=0),
            showlegend=False,
            xaxis=dict(title=f"← {first_name}   side score   {second_name} →",
                       range=[-2.1, 2.1], zeroline=False, showgrid=False),
            yaxis=dict(categoryorder="array", categoryarray=order[::-1],
                       title="", gridcolor="#E8ECF2"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            updatemenus=[dict(
                type="buttons", showactive=False, x=0, y=-0.05, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=300, redraw=True),
                                          fromcurrent=True, transition=dict(duration=150))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate", transition=dict(duration=0))]),
                ],
            )],
            sliders=[dict(
                x=0.12, len=0.88, y=-0.05, yanchor="top",
                currentvalue=dict(prefix="Window ends: ", font=dict(size=13)),
                steps=[dict(
                    method="animate", label=step_labels[i],
                    args=[[f.name], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
                ) for i, f in enumerate(frames)],
            )],
        )
        fig.add_vline(x=0, line_width=1, line_color="#1A1A1A")
        st.plotly_chart(fig, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# C · Where everyone stands
# ─────────────────────────────────────────────────────────────────────────────

def _standing(data: Data) -> None:
    st.divider()
    st.subheader("C · Where everyone stands")
    st.markdown(
        "For every country we measure its **side score**: stance toward the second party minus "
        "stance toward the first. Positive = leans second, negative = leans first, near zero = no "
        "clear side. Stance = (cooperation − conflict) / (cooperation + conflict), from −1 to +1."
    )

    have = mt.conflicts_available()
    if not have:
        return

    c_ctl, p_ctl = st.columns([2, 3], gap="large")
    with c_ctl:
        cname = st.selectbox(
            "Conflict", have,
            index=have.index("Russia-Ukraine") if "Russia-Ukraine" in have else 0,
            key="c_conflict",
        )
    first_name, second_name = cname.split("-", 1)
    ids = dict(zip(data.countries.name, data.countries.country_id))
    first_id, second_id = ids[first_name], ids[second_name]

    qs = mt.quarters_available(cname)
    with p_ctl:
        q_start, q_end = st.select_slider(
            "Period", options=qs, value=(qs[0], qs[-1]), key="c_period",
            help="Narrow to see whether countries moved between sides over time.",
        )
    chosen_q = tuple(q for q in qs if q_start <= q <= q_end)
    if q_end == qs[-1]:
        st.caption(f"The last quarter ({qs[-1]}) is incomplete — data stops 18 October 2026.")

    sc = side_scores(cname, first_id, second_id, chosen_q, 3, False)
    sc["name"] = sc.actor_id.map(data.name_of)
    sc = sc[~sc.actor_id.isin([first_id, second_id])]

    your_words("side_all")
    show_breakdown = st.checkbox(
        "Add each country's words-only and actions-only position on top of its bar",
        key="c_breakdown",
        help="The bar is one combined score. A country with friendly words and hostile "
             "actions (or the reverse) can average out to a bar that looks moderate - "
             "turn this on to see whether that is happening.",
    )
    fig, n = side_chart(sc, "side_all", first_name, second_name, "all", breakdown=show_breakdown)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{n} countries have at least 3 events toward both parties."
              + (" Tick marks: each country's words-only (purple) and actions-only (orange) "
                 "side score - where they land on opposite sides of the bar, or far from it, "
                 "the combined score is hiding a words/actions split."
                 if show_breakdown else ""))

    with st.expander("How is the side score calculated?"):
        st.markdown(
            f"For each country A and party T, we count events **A → T** in the period. "
            f"Stance(A→T) = (cooperation − conflict) / (cooperation + conflict), from −1 to +1. "
            f"**Side score** = stance toward **{second_name}** minus stance toward **{first_name}** "
            f"(range −2 to +2). A country only gets a score if it has ≥ 3 events toward **both** "
            f"parties. Staying out is not the same as being neutral."
        )


# ─────────────────────────────────────────────────────────────────────────────
# D · Words versus actions
# ─────────────────────────────────────────────────────────────────────────────

def _verbal_vs_material(data: Data) -> None:
    st.divider()
    st.subheader("D · Words versus actions")
    st.markdown(
        "GENOME distinguishes what countries **say** (verbal events) from what they **do** (material events). "
        "This chart places every country on a two-axis plane: verbal stance toward a party (horizontal) "
        "against material stance (vertical). "
        "If words and actions were the same, all dots would cluster on the diagonal. "
        "The interesting cases are the off-diagonal quadrants — friendly words with hostile actions, "
        "or the reverse."
    )

    have = mt.conflicts_available()
    if not have:
        return

    ctl1, ctl2 = st.columns([2, 3], gap="large")
    with ctl1:
        cname = st.selectbox(
            "Conflict", have,
            index=have.index("Russia-Ukraine") if "Russia-Ukraine" in have else 0,
            key="d_conflict",
        )
    first_name, second_name = cname.split("-", 1)
    ids = dict(zip(data.countries.name, data.countries.country_id))
    first_id, second_id = ids[first_name], ids[second_name]

    qs = mt.quarters_available(cname)
    with ctl2:
        q_start, q_end = st.select_slider(
            "Period", options=qs, value=(qs[0], qs[-1]), key="d_period",
        )
    chosen_q = tuple(q for q in qs if q_start <= q <= q_end)

    DEAD = 0.2
    plot_col, ctl_col = st.columns([3, 1.2], gap="large")

    with ctl_col:
        st.markdown("##### Filter")
        min_n = st.slider(
            "Min events per dot", 3, 15, 5, key="d_min_n",
            help="A dot appears only if the country has at least this many verbal events AND "
                 "material events toward that party.",
        )
        show_toward = st.radio(
            "Show toward", ["Both parties", first_name, second_name], key="d_toward",
        )
        top_n = st.slider("Name top players", 0, 20, 10, key="d_topn")

    sc = side_scores(cname, first_id, second_id, chosen_q, min_n, False)
    sc = sc.assign(name=sc.actor_id.map(data.name_of))
    sc = sc[~sc.actor_id.isin([first_id, second_id])]
    all_names = sorted(sc["name"].dropna().unique())

    with ctl_col:
        picked = st.multiselect(
            "Highlight", all_names, placeholder="Type to search…", key="d_pick",
        )

    rows = []
    for tag, party in (("first", first_name), ("second", second_name)):
        d = sc.dropna(subset=[f"verbal_{tag}", f"material_{tag}"])
        for r in d.itertuples():
            rows.append({
                "Country": r.name, "Toward": party,
                "Words": getattr(r, f"verbal_{tag}"),
                "Actions": getattr(r, f"material_{tag}"),
                "Word events":   int(getattr(r, f"n_verbal_{tag}")),
                "Action events": int(getattr(r, f"n_material_{tag}")),
            })
    q_all = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Country", "Toward", "Words", "Actions", "Word events", "Action events"])

    q = q_all if show_toward == "Both parties" else q_all[q_all.Toward == show_toward].copy()

    if q.empty:
        with plot_col:
            st.info("No dots with these settings. Lower the minimum or show both parties.")
        return

    q = q.copy()
    q["Pattern"] = [mt.pattern(w, a, DEAD) for w, a in zip(q.Words, q.Actions)]
    q["Events"]  = q["Word events"] + q["Action events"]
    is_mism = q.Pattern.isin([mt.PATTERNS["+-"], mt.PATTERNS["-+"]])
    volume  = q.groupby("Country").Events.sum().sort_values(ascending=False)
    big     = set(volume.head(top_n).index)
    q["named"]  = q.Country.isin(big) | is_mism | q.Country.isin(picked)
    q["picked"] = q.Country.isin(picked)

    rng = np.random.default_rng(0)
    q["x"] = q.Words   + rng.uniform(-0.025, 0.025, len(q))
    q["y"] = q.Actions + rng.uniform(-0.025, 0.025, len(q))
    q["size"] = (6 + 1.1 * np.sqrt(q.Events)).clip(6, 30)

    fig = go.Figure()
    # Subtle shading on the mismatch quadrants
    for x0, x1, y0, y1 in [(0, 1.2, -1.2, 0), (-1.2, 0, 0, 1.2)]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor="rgba(198,40,40,0.06)", line_width=0, layer="below")

    for party, colr in ((first_name, FIRST_COLOR), (second_name, SECOND_COLOR)):
        dd = q[q.Toward == party]
        if dd.empty:
            continue
        dd_m = dd.Pattern.isin([mt.PATTERNS["+-"], mt.PATTERNS["-+"]])
        fig.add_trace(go.Scatter(
            x=dd.x, y=dd.y, mode="markers+text", name=f"→ {party}",
            text=[c if n else "" for c, n in zip(dd.Country, dd.named)],
            textposition="top center", textfont=dict(size=10, color="#1A1A1A"),
            marker=dict(
                size=dd["size"], color=colr,
                opacity=[0.95 if (m or p) else 0.55 for m, p in zip(dd_m, dd.picked)],
                line=dict(
                    color=["#1A1A1A" if p else "white" for p in dd.picked],
                    width=[2.5 if p else 1 for p in dd.picked],
                ),
            ),
            customdata=dd[["Country", "Toward", "Words", "Actions", "Word events", "Action events"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b> → %{customdata[1]}<br>"
                "words %{customdata[2]:.2f}  (%{customdata[4]} events)<br>"
                "actions %{customdata[3]:.2f}  (%{customdata[5]} events)"
                "<extra></extra>"
            ),
        ))

    for x, y, txt, anc, col in [
        ( 1.18,  1.12, "Friendly words,<br>friendly actions",        "right", SECOND_COLOR),
        (-1.18, -1.12, "Hostile words,<br>hostile actions",          "left",  FIRST_COLOR),
        ( 1.18, -1.12, "Friendly words,<br><b>hostile actions</b>",  "right", FIRST_COLOR),
        (-1.18,  1.12, "Hostile words,<br><b>friendly actions</b>",  "left",  FIRST_COLOR),
    ]:
        fig.add_annotation(
            x=x, y=y, text=txt, showarrow=False, xanchor=anc,
            yanchor="top" if y > 0 else "bottom",
            font=dict(size=11, color=col),
        )
    fig.add_hline(y=0, line_width=1, line_color="#1A1A1A")
    fig.add_vline(x=0, line_width=1, line_color="#1A1A1A")
    fig.update_layout(
        height=600, margin=dict(l=0, r=10, t=10, b=0),
        xaxis=dict(title="Words:  hostile ←  → friendly",
                   range=[-1.2, 1.2], zeroline=False, showgrid=False),
        yaxis=dict(title="Actions:  hostile ←  → friendly",
                   range=[-1.2, 1.2], zeroline=False, gridcolor="#E8ECF2"),
        legend=dict(orientation="h", y=1.06),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )

    consistent = q.Pattern.isin([mt.PATTERNS["++"], mt.PATTERNS["--"]]).sum()
    mismatch   = int(is_mism.sum())
    mixed      = (q.Pattern == mt.PATTERNS["mixed"]).sum()

    with plot_col:
        st.markdown(
            f"**{len(q)} dots** · {consistent} consistent · **{mismatch} mismatches** · {mixed} mixed/neutral"
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each dot = one country toward one party. Bigger dot = more events. "
            "Named: biggest players, every mismatch, and any highlighted country. "
            "Dots nudged slightly apart; hover for exact values."
        )

    # Mismatch tables
    if mismatch > 0:
        mism = q[is_mism].copy()
        mism["Gap"] = (mism.Actions - mism.Words).abs()
        rows_pat = mism[mism.Pattern == mt.PATTERNS["+-"]].sort_values("Gap", ascending=False)
        if not rows_pat.empty:
            st.markdown(f"##### {mt.PATTERNS['+-']}")
            st.dataframe(
                rows_pat[["Country", "Toward", "Words", "Actions",
                           "Word events", "Action events"]].round(2),
                hide_index=True, use_container_width=True,
                height=min(300, 40 + 35 * len(rows_pat)),
            )

    # Key takeaway
    st.info(
        f"**Takeaway — the verbal and material world are separate.** "
        f"Of the {len(q)} country–party pairs shown, **{mismatch}** show a clear mismatch "
        "between words and actions. This is precisely why we treat the two dimensions separately "
        "throughout this analysis: a stance built only on statements can point in a completely "
        "different direction from one built on actual events."
    )

    st.caption(
        "**Note:** in GENOME a *request* counts as a hostile word. A country asking another for "
        "help (e.g. Ukraine asking the US) will show 'hostile words' next to 'friendly actions'. "
        "Check the event counts before drawing conclusions."
    )
