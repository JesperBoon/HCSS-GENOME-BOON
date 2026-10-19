"""
build_monthly_counts.py — download monthly event counts per country, event type.

Uses ONLY free, unauthenticated count queries (the `total` field of /events page 1),
so it costs no credits. Role = actor. Output: data/processed/monthly_counts.csv
columns: country_id, month, event_type, n.  Resumable: finished countries are skipped.

Run from project root:  .venv/bin/python scripts/build_monthly_counts.py
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

B = "https://genome.app.hcss.nl/api"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "monthly_counts.csv"
MIN_EVENTS = 200            # only countries with at least this many events as actor
START = "2024-01"           # 2021-2023 holds < 500 events in total
TYPES = ["AGREE", "CONSULT", "SUPPORT", "CONCEDE", "COOPERATE", "AID", "RETREAT",
         "REQUEST", "ACCUSE", "REJECT", "THREATEN", "PROTEST", "SANCTION",
         "MOBILIZE", "COERCE", "ASSAULT"]

s = requests.Session()
s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))


def total(params):
    for attempt in range(4):
        try:
            return s.get(f"{B}/events", params={**params, "page": 1}, timeout=60).json()["total"]
        except Exception:
            time.sleep(1 + attempt)
    return None


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    countries = s.get(f"{B}/countries").json()["countries"]
    last = s.get(f"{B}/events/time-range").json()["latest"]
    months = pd.period_range(START, last[:7], freq="M")

    with ThreadPoolExecutor(12) as ex:
        sizes = dict(zip([c["id"] for c in countries],
                         ex.map(lambda c: total({"actor_country_id": c["id"]}), countries)))
    todo = [cid for cid, n in sizes.items() if n and n >= MIN_EVENTS]
    done = set(pd.read_csv(OUT).country_id) if OUT.exists() else set()
    todo = [c for c in todo if c not in done]
    print(f"{len(todo)} countries to fetch ({len(done)} already done)")

    for i, cid in enumerate(todo, 1):
        jobs = [(t, m) for t in TYPES for m in months]
        def run(j):
            t, m = j
            return total({"actor_country_id": cid, "event_type": t,
                          "event_date_from": str(m.start_time.date()),
                          "event_date_to": str(m.end_time.date())})
        with ThreadPoolExecutor(16) as ex:
            ns = list(ex.map(run, jobs))
        df = pd.DataFrame({"country_id": cid, "month": [str(m) for _, m in jobs],
                           "event_type": [t for t, _ in jobs], "n": ns})
        if df.n.isna().any():
            print(f"  country {cid}: {df.n.isna().sum()} failed queries, skipped (will retry on rerun)")
            continue
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"[{i}/{len(todo)}] country {cid} done", flush=True)


if __name__ == "__main__":
    main()
