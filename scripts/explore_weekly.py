"""Feasibility spike: weekly stance counts for ten big players in Russia vs Ukraine. Free queries only."""
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
B = "https://genome.app.hcss.nl/api/events"
ids = {c["name"]: c["id"] for c in requests.get("https://genome.app.hcss.nl/api/countries").json()["countries"]}
actors = ["USA", "Germany", "UK", "France", "Poland", "China", "India", "Iran", "North Korea", "Türkiye"]
targets = ["Russia", "Ukraine"]
weeks = pd.period_range("2024-01-01", "2026-10-18", freq="W-SUN")
cats = {"coop": "COOPERATION", "conf": "CONFLICT"}
jobs = [(a, t, w, k) for a in actors for t in targets for w in weeks for k in cats]
def f(j):
    a, t, w, k = j
    p = {"actor_country_id": ids[a], "recipient_country_id": ids[t], "category": cats[k], "page": 1,
         "event_date_from": str(w.start_time.date()), "event_date_to": str(w.end_time.date())}
    for _ in range(3):
        try: return requests.get(B, params=p, timeout=60).json()["total"]
        except Exception: time.sleep(1)
t0 = time.time()
with ThreadPoolExecutor(16) as ex: n = list(ex.map(f, jobs))
d = pd.DataFrame(jobs, columns=["actor", "target", "week", "k"]); d["n"] = n
d["week"] = d.week.astype(str)
print(f"{len(jobs)} queries in {time.time()-t0:.0f}s ({len(jobs)/(time.time()-t0):.0f}/s), failed {d.n.isna().sum()}")
d.to_csv("data/cache/weekly_spike.csv", index=False)
