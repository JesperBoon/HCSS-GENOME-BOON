"""
build_dyad_counts.py — quarterly stance counts of every country toward both sides of a conflict.

FREE: only unauthenticated page-1 count queries (the `total` field). Spends no credits.

For a conflict (X, Y) and every tracked country A, for each target T in {X, Y} (A != T)
and each quarter, count events A -> T:
    coop_all, conf_all           by `category`
    AID, COOPERATE, RETREAT      material cooperation (event_type)
    ASSAULT, MOBILIZE, COERCE, SANCTION, PROTEST   material conflict (event_type; PROTEST
                                           counts as material conflict, as in the GENOME paper, Table 4)
    AGREE, CONSULT, SUPPORT, CONCEDE, REQUEST, ACCUSE, REJECT, THREATEN   verbal (event_type)
All 16 event types are stored separately, so events can be weighted by type (severity);
coop_all / conf_all are kept as category totals. 18 queries per country-target-quarter.

Output: data/processed/dyad_counts.csv  (conflict, actor_id, target_id, quarter, key, n).
Resumable per (conflict, actor, key): only missing keys are fetched, so adding a new
event type to KEYS and rerunning fills just that type.

Usage (project root):  .venv/bin/python scripts/build_dyad_counts.py "Russia" "Ukraine"
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

B = "https://genome.app.hcss.nl/api"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "dyad_counts.csv"
START = "2024Q1"

MATERIAL = ["AID", "COOPERATE", "RETREAT", "ASSAULT", "MOBILIZE", "COERCE", "SANCTION", "PROTEST"]
VERBAL = ["AGREE", "CONSULT", "SUPPORT", "CONCEDE", "REQUEST", "ACCUSE", "REJECT", "THREATEN"]
KEYS = {"coop_all": {"category": "COOPERATION"}, "conf_all": {"category": "CONFLICT"},
        **{t: {"event_type": t} for t in MATERIAL + VERBAL}}

s = requests.Session()
s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=32))


def total(params):
    for attempt in range(4):
        try:
            return s.get(f"{B}/events", params={**params, "page": 1}, timeout=60).json()["total"]
        except Exception:
            time.sleep(1 + attempt)
    return None


def main(x_name, y_name):
    names = {c["name"]: c["id"] for c in s.get(f"{B}/countries").json()["countries"]}
    x, y = names[x_name], names[y_name]
    label = f"{x_name}-{y_name}"
    tracked = sorted(pd.read_csv(ROOT / "data/processed/monthly_counts.csv").country_id.unique())
    last = s.get(f"{B}/events/time-range").json()["latest"]
    quarters = pd.period_range(START, last[:7], freq="Q")

    done = set()   # (actor_id, key) pairs already stored for this conflict
    if OUT.exists():
        d = pd.read_csv(OUT, usecols=["conflict", "actor_id", "key"])
        d = d[d.conflict == label].drop_duplicates()
        done = set(zip(d.actor_id, d.key))
    todo = [a for a in tracked if any((a, k) not in done for k in KEYS)]
    print(f"{label}: {len(todo)} actors to fetch, {len(quarters)} quarters", flush=True)

    for i, a in enumerate(todo, 1):
        keys = [k for k in KEYS if (a, k) not in done]
        jobs = [(t, q, k) for t in (x, y) if t != a for q in quarters for k in keys]

        def run(j):
            t, q, k = j
            return total({"actor_country_id": a, "recipient_country_id": t, **KEYS[k],
                          "event_date_from": str(q.start_time.date()),
                          "event_date_to": str(q.end_time.date())})
        with ThreadPoolExecutor(16) as ex:
            ns = list(ex.map(run, jobs))
        df = pd.DataFrame({"conflict": label, "actor_id": a, "target_id": [j[0] for j in jobs],
                           "quarter": [str(j[1]) for j in jobs], "key": [j[2] for j in jobs], "n": ns})
        if df.n.isna().any():
            print(f"  actor {a}: {df.n.isna().sum()} failed queries, skipped (rerun to retry)")
            continue
        df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"[{i}/{len(todo)}] actor {a} done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
