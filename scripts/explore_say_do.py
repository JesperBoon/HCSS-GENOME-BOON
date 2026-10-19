"""Feasibility spike: monthly event-type counts per actor country, free queries only."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

B = "https://genome.app.hcss.nl/api/events"
TYPES = ["AGREE","CONSULT","SUPPORT","CONCEDE","COOPERATE","AID","RETREAT",
         "REQUEST","ACCUSE","REJECT","THREATEN","PROTEST","SANCTION","MOBILIZE","COERCE","ASSAULT"]
COUNTRIES = ["Russia","Ukraine","Israel","Iran","USA","China"]
# Text filters are fuzzy ("US" matches far more than the USA); ids are exact.
IDS = {c["name"]: c["id"] for c in requests.get("https://genome.app.hcss.nl/api/countries").json()["countries"]}
months = pd.period_range("2024-01", "2026-09", freq="M")

def job(a):
    c, t, m = a
    p = {"actor_country_id": IDS[c], "event_type": t, "page": 1,
         "event_date_from": str(m.start_time.date()), "event_date_to": str(m.end_time.date())}
    for _ in range(3):
        try:
            return (c, t, str(m), requests.get(B, params=p, timeout=60).json()["total"])
        except Exception:
            time.sleep(1)
    return (c, t, str(m), None)

jobs = [(c, t, m) for c in COUNTRIES for t in TYPES for m in months]
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(job, jobs))
df = pd.DataFrame(rows, columns=["country", "event_type", "month", "n"])
print(f"{len(jobs)} requests in {time.time()-t0:.0f}s, failed: {df.n.isna().sum()}")
df.to_csv("data/cache/say_do_spike.csv", index=False)
