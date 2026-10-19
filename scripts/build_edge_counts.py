"""
build_edge_counts.py — who does what to whom, between countries: the edges of the support-chain graph.

FREE: uses a free-only API client (src/genome_api.py). Any request that needs the API key, such as
page 2 of /events, raises PaidRequestBlocked before it is sent. No credits can be spent.

For every ordered pair of countries (actor -> recipient) and every event type of the two layers
    material  (taxonomy: AID, COOPERATE, RETREAT, ASSAULT, SANCTION, COERCE, MOBILIZE, PROTEST)  down to weeks
    words     (AGREE, CONSULT, SUPPORT, CONCEDE, REQUEST, ACCUSE, REJECT, THREATEN)              down to months
it takes the exact count for the whole period and splits it adaptively (quarter, month, week) only
while a window holds more than 250 events. A window with at most 250 events is closed and its events
are taken from the free page 1. Organisations (EU, NATO, UN) have no country id in GENOME and cannot
be filtered exactly, so only countries are in the graph.

Default is a DRY RUN: it measures a sample, prints the estimated number of requests and minutes, and
stops. Add --run to build.

Output (data/processed/edges/, small enough to commit):
    leaf_counts.csv   layer, event_type, category, actor_id, recipient_id, window_start, window_end, count, source
    events.csv        event_id, event_date, event_type, category, intensity, article_count
    event_edges.csv   event_id, actor_id, recipient_id    (pairs inside the country set)
    done_cells.csv    which (layer, type, actor, recipient) cells are finished  -> the script is resumable
    meta.json         parameters, the country set and the date of the run

Usage (project root):
    .venv/bin/python scripts/build_edge_counts.py                      # dry run
    .venv/bin/python scripts/build_edge_counts.py --run                # build
    .venv/bin/python scripts/build_edge_counts.py --countries 40 --extra Belarus,Taiwan --run
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import taxonomy as tx                                                    # noqa: E402
from edge_counts import LIMIT, Window, adaptive_windows, compact_event   # noqa: E402
from genome_api import GenomeClient                                      # noqa: E402

OUT = ROOT / "data" / "processed" / "edges"
FINEST = {"material": "week", "words": "month"}
TYPES = {"material": tx.MATERIAL_TYPES, "words": tx.WORD_TYPES}
LEAF_COLS = ["layer", "event_type", "category", "actor_id", "recipient_id",
             "window_start", "window_end", "count", "source"]
EVENT_COLS = ["event_id", "event_date", "event_type", "category", "intensity", "article_count"]
EDGE_COLS = ["event_id", "actor_id", "recipient_id"]
DONE_COLS = ["layer", "event_type", "actor_id", "recipient_id"]

client = GenomeClient(free_only=True)


def retry(fn, *args, **kwargs):
    for attempt in range(4):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                  # network hiccup; PaidRequestBlocked must never be retried
            if type(exc).__name__ == "PaidRequestBlocked" or attempt == 3:
                raise
            time.sleep(1 + attempt)


def count(a, r, window=None, event_type=None):
    f = {"actor_country_id": a, "recipient_country_id": r}
    if event_type:
        f["event_type"] = event_type
    if window:
        f.update(event_date_from=str(window.start), event_date_to=str(window.end))
    return retry(client.count, **f)


def country_set(n, extra, since):
    """Top-n countries by volume (monthly_counts.csv) plus the named extras. Returns {id: name}."""
    names = {c["id"]: c["name"] for c in client.countries()["countries"]}
    by_name = {v: k for k, v in names.items()}
    top = pd.read_csv(ROOT / "data/processed/monthly_counts.csv").groupby("country_id").n.sum()
    ids = top.sort_values(ascending=False).head(n).index.tolist()
    for name in extra:
        if name not in by_name:
            sys.exit(f"Unknown country {name!r}. Organisations (EU, NATO, UN) have no country id.")
        if by_name[name] not in ids:
            ids.append(by_name[name])
    return {i: names[i] for i in ids}


# ── one cell ─────────────────────────────────────────────────────────────────

def process_cell(layer, et, a, r, total0, whole, ids, fetch_events=True):
    """Adaptive windows for one (layer, type, actor, recipient) cell. Returns (leaf rows, event rows, edge rows, n_requests)."""
    requests_made = [0]

    def counter(w):
        requests_made[0] += 1
        return count(a, r, w, et)

    leaves = adaptive_windows(counter, whole, FINEST[layer], LIMIT, known_total=total0)
    leaf_rows, events, edges = [], {}, []
    for leaf in leaves:
        leaf_rows.append({"layer": layer, "event_type": et, "category": tx.category_of(et), "actor_id": a,
                          "recipient_id": r, "window_start": leaf.window.start, "window_end": leaf.window.end,
                          "count": leaf.total, "source": "events" if leaf.has_events else "count"})
        if not leaf.has_events:
            continue
        requests_made[0] += 1
        if not fetch_events:
            continue
        page = retry(client.events_page, 1, False, actor_country_id=a, recipient_country_id=r, event_type=et,
                     event_date_from=str(leaf.window.start), event_date_to=str(leaf.window.end))
        if len(page["events"]) != leaf.total:
            raise RuntimeError(f"page 1 holds {len(page['events'])} events but total says {leaf.total} "
                               f"({layer} {et} {a}->{r} {leaf.window}); new data may have arrived, rerun")
        for e in page["events"]:
            row, es = compact_event(e, ids)
            events[row["event_id"]] = row
            edges.extend(es)
    return leaf_rows, list(events.values()), edges, requests_made[0]


# ── dry run ──────────────────────────────────────────────────────────────────

def dry_run(pairs, whole, ids, sample, workers):
    rng = random.Random(1)
    picked = rng.sample(pairs, min(sample, len(pairs)))

    def one(pair):
        a, r = pair
        reqs, events = 0, 0
        for layer in ("material", "words"):
            for et in TYPES[layer]:
                reqs += 1
                t = count(a, r, whole, et)
                if t:
                    _, ev, _, n = process_cell(layer, et, a, r, t, whole, ids, fetch_events=False)
                    reqs += n
                    events += t
        return reqs, events

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = pd.DataFrame(list(ex.map(one, picked)), columns=["reqs", "events"])
    rate = (res.reqs.sum() + len(picked)) / (time.time() - t0) * 60
    n = len(pairs)
    est = res.reqs.mean() * n
    se = res.reqs.std() / len(res) ** 0.5 * n
    print(f"\nDry run on {len(res)} sampled pairs ({time.time() - t0:.0f}s, {rate:,.0f} requests/minute measured):")
    print(f"  pairs with at least one event: {n:,}")
    print(f"  estimated requests: {est + n:,.0f} ± {se:,.0f}   (+ {n:,} pair-total queries already made)")
    print(f"  estimated duration: {(est + n) / rate:,.0f} minutes at {workers} workers")
    print(f"  events that will be stored (upper bound, an event repeats across cells): "
          f"about {res.events.mean() * n:,.0f}")
    print("\nNothing was written. Add --run to build.")


# ── storage ──────────────────────────────────────────────────────────────────

def append(path, rows, cols):
    if not rows:
        return
    pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def verify(ids):
    """Every closed window must be reproduced exactly by the stored events and edges."""
    leaves = pd.read_csv(OUT / "leaf_counts.csv", parse_dates=["window_start", "window_end"])
    leaves = leaves[leaves.source == "events"]
    ev = pd.read_csv(OUT / "events.csv", parse_dates=["event_date"]).drop_duplicates("event_id")
    ed = pd.read_csv(OUT / "event_edges.csv").drop_duplicates()
    m = ed.merge(ev[["event_id", "event_date", "event_type"]], on="event_id")
    got = {}
    for (a, r, et), g in m.groupby(["actor_id", "recipient_id", "event_type"]):
        got[(a, r, et)] = g.event_date.sort_values().values
    bad = 0
    for row in leaves.itertuples():
        dates = got.get((row.actor_id, row.recipient_id, row.event_type), [])
        n = int(((dates >= row.window_start.to_datetime64()) & (dates <= row.window_end.to_datetime64())).sum()) \
            if len(dates) else 0
        if n != row.count:
            bad += 1
    print(f"Verification: {len(leaves):,} closed windows re-counted from the stored events; "
          f"{bad} do not match.")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", type=int, default=60, help="size of the country set (top by volume)")
    ap.add_argument("--extra", default="", help="comma-separated extra countries, e.g. Belarus,Taiwan")
    ap.add_argument("--since", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=150, help="pairs sampled by the dry run")
    ap.add_argument("--run", action="store_true", help="really build (default: dry run)")
    args = ap.parse_args()

    latest = date.fromisoformat(client._get("/events/time-range")["latest"])
    whole = Window(date.fromisoformat(args.since), latest)
    ids = country_set(args.countries, [x for x in args.extra.split(",") if x], args.since)
    id_list = list(ids)
    print(f"{len(ids)} countries, {whole.start} to {whole.end}; ordered pairs: {len(ids) * (len(ids) - 1):,}")

    t0 = time.time()
    all_pairs = [(a, r) for a in id_list for r in id_list if a != r]
    with ThreadPoolExecutor(args.workers) as ex:
        totals = dict(zip(all_pairs, ex.map(lambda p: count(p[0], p[1], whole), all_pairs)))
    pairs = [p for p, t in totals.items() if t > 0]
    print(f"Pair totals ({len(all_pairs):,} requests, {time.time() - t0:.0f}s): "
          f"{len(pairs):,} pairs have at least one event.")

    if not args.run:
        dry_run(pairs, whole, set(id_list), args.sample, args.workers)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if (OUT / "done_cells.csv").exists():
        d = pd.read_csv(OUT / "done_cells.csv")
        done = set(zip(d.layer, d.event_type, d.actor_id, d.recipient_id))
    cells = [(layer, et, a, r) for (a, r) in pairs for layer in ("material", "words") for et in TYPES[layer]
             if (layer, et, a, r) not in done]
    seen_events = set(pd.read_csv(OUT / "events.csv", usecols=["event_id"]).event_id) \
        if (OUT / "events.csv").exists() else set()
    seen_edges = set()
    (OUT / "meta.json").write_text(json.dumps({
        "built": str(pd.Timestamp.now()), "since": args.since, "latest": str(latest),
        "countries": {str(k): v for k, v in ids.items()}, "limit": LIMIT, "finest": FINEST}, indent=1))
    print(f"{len(cells):,} cells to do ({len(done):,} already done). Building...")

    def work(cell):
        layer, et, a, r = cell
        t = count(a, r, whole, et)
        if t == 0:
            return cell, [], [], [], 1
        lr, ev, ed, n = process_cell(layer, et, a, r, t, whole, set(id_list))
        return cell, lr, ev, ed, n + 1

    buf = {"leaf": [], "ev": [], "ed": [], "done": []}
    n_req, t0 = 0, time.time()

    def flush():
        append(OUT / "leaf_counts.csv", buf["leaf"], LEAF_COLS)
        append(OUT / "events.csv", buf["ev"], EVENT_COLS)
        append(OUT / "event_edges.csv", buf["ed"], EDGE_COLS)
        append(OUT / "done_cells.csv", buf["done"], DONE_COLS)         # last: a cell counts as done only after its data
        for k in buf:
            buf[k] = []

    with ThreadPoolExecutor(args.workers) as ex:
        for i, (cell, lr, ev, ed, n) in enumerate(ex.map(work, cells), 1):
            n_req += n
            buf["leaf"].extend(lr)
            buf["ev"].extend(e for e in ev if e["event_id"] not in seen_events)
            seen_events.update(e["event_id"] for e in ev)
            buf["ed"].extend(e for e in ed if e not in seen_edges)
            seen_edges.update(ed)
            buf["done"].append(dict(zip(DONE_COLS, cell)))
            if i % 500 == 0:
                flush()
                el = time.time() - t0
                print(f"  {i:,}/{len(cells):,} cells, {n_req:,} requests, {el / 60:.1f} min, "
                      f"~{el / i * (len(cells) - i) / 60:.1f} min to go", flush=True)
    flush()
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes, {n_req:,} requests, 0 credits.")
    bad = verify(set(id_list))
    for f in ("leaf_counts", "events", "event_edges"):
        n = sum(1 for _ in open(OUT / f"{f}.csv")) - 1
        print(f"  {f}.csv: {n:,} rows, {(OUT / f'{f}.csv').stat().st_size / 1e6:.1f} MB")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
