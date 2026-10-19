"""Pages that are not built yet: only a place for the author's own words."""

import streamlit as st

from views.common import your_words

TITLES = {"6": ("Step 6 · Escalation ladder", ["escalation"]),
          "7": ("Step 7 · Limits & next steps", ["limits", "next_steps"])}


def render(page: str) -> None:
    title, keys = TITLES[page[0]]
    st.title(title)
    for k in keys:
        your_words(k)
    st.info("Not built yet.")
