"""
find_conflict_pairs.py — which country pairs are in the most hostile mutual conflict?

FREE: only unauthenticated count queries (the `total` field), no credits.
Stage 1: for every ordered pair of the 87 tracked countries, count CONFLICT events A -> B
         since 2024-01-01 (7,482 queries; cached in data/cache/pairs_stage1.csv).
Stage 2: for the 25 pairs with the most mutual conflict, also count COOPERATION events in
         both directions, to get the share of their interactions that is hostile.
Output:  data/processed/conflict_pairs.csv

Run (project root):  .venv/bin/python scripts/find_conflict_pairs.py
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

B = "https://genome.app.hcss.nl/api"
ROOT = Path(__file__).resolve().parent.parent
STAGE1 = ROOT / "data" / "cache" / "pairs_stage1.csv"
OUT = ROOT / "data" / "processed" / "conflict_pairs.csv"
SINCE = "2024-01-01"
TOP = 25


def total(a, b, category):
    for attempt in range(4):
        try:
            return requests.get(f"{B}/events", params={
                "actor_country_id": a, "recipient_country_id": b, "page": 1,
                "category": category, "event_date_from": SINCE}, timeout=60).json()["total"]
        except Exception:
            time.sleep(1 + attempt)


def main():
    names = {c["id"]: c["name"] for c in requests.get(f"{B}/countries").json()["countries"]}
    if STAGE1.exists():
        d = pd.read_csv(STAGE1)
    else:
        ids = sorted(pd.read_csv(ROOT / "data/processed/monthly_counts.csv").country_id.unique())
        jobs = [(a, b) for a in ids for b in ids if a != b]
        with ThreadPoolExecutor(16) as ex:
            n = list(ex.map(lambda j: total(*j, "CONFLICT"), jobs))
        d = pd.DataFrame(jobs, columns=["a", "b"]).assign(conf=n)
        STAGE1.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(STAGE1, index=False)

    m = d.merge(d.rename(columns={"a": "b", "b": "a", "conf": "conf_rev"}), on=["a", "b"])
    m = m[m.a < m.b].assign(mutual=lambda x: x[["conf", "conf_rev"]].min(axis=1))
    top = m.sort_values("mutual", ascending=False).head(TOP).copy()

    with ThreadPoolExecutor(16) as ex:
        top["coop"] = list(ex.map(lambda r: total(r[0], r[1], "COOPERATION"), zip(top.a, top.b)))
        top["coop_rev"] = list(ex.map(lambda r: total(r[1], r[0], "COOPERATION"), zip(top.a, top.b)))
    top["first"], top["second"] = top.a.map(names), top.b.map(names)
    top["conflict_events"] = top.conf + top.conf_rev
    top["coop_events"] = top.coop + top.coop_rev
    top["hostile_share"] = top.conflict_events / (top.conflict_events + top.coop_events)
    top.rename(columns={"conf": "conf_a_to_b", "conf_rev": "conf_b_to_a"}).to_csv(OUT, index=False)
    print(top[["first", "second", "mutual", "hostile_share"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
