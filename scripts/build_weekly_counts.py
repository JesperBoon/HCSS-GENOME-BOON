"""
build_weekly_counts.py — weekly event counts per event type for the biggest players in a conflict.

FREE: only unauthenticated page-1 count queries (the `total` field). Spends no credits.

For a conflict (X, Y) the 30 players with the most events toward BOTH parties are picked from the
quarterly data (data/processed/dyad_counts.csv; ranked by the smaller of their events toward X and
toward Y, parties themselves excluded). For each player A, target T in {X, Y} and ISO week
(Monday to Sunday, from 2024-01-01) it counts the events A -> T of each of the 16 event types.

Output: data/processed/weekly/<X>-<Y>.csv with columns actor_id, target_id, week_start, key, n.
Zero counts are not stored (the reader fills them in). A row with key `_done` marks a finished
player, so the script is resumable.

Usage (project root):  .venv/bin/python scripts/build_weekly_counts.py Russia Ukraine
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

B = "https://genome.app.hcss.nl/api"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed" / "weekly"
START = "2024-01-01"      # a Monday
TOP_N = 30
TYPES = ["AGREE", "CONSULT", "SUPPORT", "CONCEDE", "REQUEST", "ACCUSE", "REJECT", "THREATEN",
         "AID", "COOPERATE", "RETREAT", "ASSAULT", "MOBILIZE", "COERCE", "SANCTION", "PROTEST"]

s = requests.Session()
s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=48))


def total(params):
    for attempt in range(4):
        try:
            return s.get(f"{B}/events", params={**params, "page": 1}, timeout=60).json()["total"]
        except Exception:
            time.sleep(1 + attempt)
    return None


def top_players(label, x, y):
    d = pd.read_csv(ROOT / "data/processed/dyad_counts.csv")
    d = d[(d.conflict == label) & d.key.isin(["coop_all", "conf_all"])]
    ev = d.groupby(["actor_id", "target_id"]).n.sum().unstack("target_id").fillna(0)
    ev = ev[~ev.index.isin([x, y])]
    rank = pd.DataFrame({"floor": ev[[x, y]].min(axis=1), "sum": ev[[x, y]].sum(axis=1)})
    return rank.sort_values(["floor", "sum"], ascending=False).head(TOP_N).index.tolist()


def main(x_name, y_name):
    names = {c["name"]: c["id"] for c in s.get(f"{B}/countries").json()["countries"]}
    x, y = names[x_name], names[y_name]
    label = f"{x_name}-{y_name}"
    out = OUT_DIR / f"{label}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    last = pd.Timestamp(s.get(f"{B}/events/time-range").json()["latest"])
    weeks = pd.date_range(START, last, freq="W-MON")

    done = set(pd.read_csv(out, usecols=["actor_id", "key"]).query("key == '_done'").actor_id) if out.exists() else set()
    todo = [a for a in top_players(label, x, y) if a not in done]
    print(f"{label}: {len(todo)} players to fetch, {len(weeks)} weeks", flush=True)

    for i, a in enumerate(todo, 1):
        jobs = [(t, w, k) for t in (x, y) for w in weeks for k in TYPES]

        def run(j):
            t, w, k = j
            return total({"actor_country_id": a, "recipient_country_id": t, "event_type": k,
                          "event_date_from": str(w.date()), "event_date_to": str((w + pd.Timedelta(days=6)).date())})
        with ThreadPoolExecutor(32) as ex:
            ns = list(ex.map(run, jobs))
        df = pd.DataFrame({"actor_id": a, "target_id": [j[0] for j in jobs],
                           "week_start": [str(j[1].date()) for j in jobs], "key": [j[2] for j in jobs], "n": ns})
        if df.n.isna().any():
            print(f"  player {a}: {df.n.isna().sum()} failed queries, skipped (rerun to retry)", flush=True)
            continue
        df = df[df.n > 0]
        marker = pd.DataFrame({"actor_id": [a], "target_id": [x], "week_start": [str(weeks[0].date())],
                               "key": ["_done"], "n": [0]})
        pd.concat([df, marker]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[{i}/{len(todo)}] player {a} done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
