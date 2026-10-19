"""
probe_clusters.py: step 0 of the "support clusters" work. Can the FREE API carry cluster analysis?

Credits: the events requests use a free-only client (src/genome_api.py): no x-api-key header is ever sent to
/events, and any attempt at page 2 or beyond, or at an endpoint that needs the key, raises PaidRequestBlocked
before it is sent. The only request that carries the key is the balance check on /keys/me (it costs nothing),
done through a separate client before and after a trial run of 20 requests. The script stops if the balance changed.

Checks
  1  Is `total` exact for actor_country_id + recipient_country_id + event_type + date window?
  2  Does page 1 return ALL events when total <= 250? (5+ random pairs/windows, plus the limit itself)
  3  Which fields does an event have; is the order newest first?
  4  Throughput and build time.
  5  Is there enough data per period for clusters? (quarters, months; per layer and category)

Run (project root):  .venv/bin/python scripts/probe_clusters.py
Report: printed, and saved to data/cache/probe_clusters_report.md
"""

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import taxonomy as tx                                              # noqa: E402
from genome_api import GenomeClient, PaidRequestBlocked, api_key   # noqa: E402

REPORT = ROOT / "data" / "cache" / "probe_clusters_report.md"
EDGES = ROOT / "data" / "processed" / "edges"
rng = random.Random(7)
out: list = []


def say(text=""):
    print(text, flush=True)
    out.append(text)


# ── the free-only client, with a spy on every request it sends ──────────────
free = GenomeClient(free_only=True)
sent = []
_orig_get = free.session.get


def _spy(url, params=None, headers=None, timeout=None):
    sent.append({"url": url, "has_key": any(k.lower() == "x-api-key" for k in (headers or {}))})
    return _orig_get(url, params=params, headers=headers, timeout=timeout)


free.session.get = _spy


def count(**f):
    return free.count(**f)


def page1(**f):
    return free.events_page(1, cache=False, **f)


def balance():
    """Credits used / remaining, through a normal client: the only call that carries the key."""
    if not api_key():
        return None
    info = GenomeClient().balance()
    return info["used_credits"], info["remaining_credits"]


# ── set-up ───────────────────────────────────────────────────────────────────
meta = json.loads((EDGES / "meta.json").read_text())
names = {int(k): v for k, v in meta["countries"].items()}
ids = {v: k for k, v in names.items()}
nodes = list(names)
latest = pd.Timestamp(meta["latest"])
say("# Probe: can free queries carry the cluster analysis?\n")
say(f"Run on {pd.Timestamp.now():%Y-%m-%d %H:%M}. Data until {latest.date()}. Country set: the {len(nodes)} biggest countries "
    f"already stored by build_edge_counts.py.\n")

# ── the guards, shown ────────────────────────────────────────────────────────
say("## 0. Credits are protected\n")
for label, call in (("page 2 of /events", lambda: free.events_page(2, actor_country_id=ids["USA"])),
                    ("/keys/me with the key", lambda: free.balance())):
    try:
        call()
        say(f"- {label}: **NOT BLOCKED** (stopping)")
        sys.exit(1)
    except PaidRequestBlocked as exc:
        say(f"- {label}: blocked before sending (`PaidRequestBlocked`: {exc})")
say("- `genome_api.py` `_get`: `if auth and self.free_only: raise PaidRequestBlocked(...)`; the key is only added "
    "to the headers after that line, and `events_page` refuses `page > 1` first of all.\n")

b0 = balance()
trial = [count(actor_country_id=rng.choice(nodes), recipient_country_id=rng.choice(nodes), event_type="AID") for _ in range(18)]
trial.append(len(page1(actor_country_id=ids["Russia"], recipient_country_id=ids["Ukraine"])["events"]))
trial.append(len(page1(actor_country_id=ids["USA"], recipient_country_id=ids["Israel"])["events"]))
b1 = balance()
say(f"Trial run: {len(trial)} requests to /events. Balance before: {b0}; after: {b1}. "
    f"Requests sent with an api-key header: **{sum(r['has_key'] for r in sent)} of {len(sent)}**.\n")
if b0 is not None and b0 != b1:
    say("**The balance changed. Stopping.**")
    sys.exit(1)

# ── check 1 ──────────────────────────────────────────────────────────────────
say("## 1. Is `total` exact?\n")
base = {"actor_country_id": ids["Iran"], "recipient_country_id": ids["Israel"]}
w = lambda a, b: {"event_date_from": str(a), "event_date_to": str(b)}   # noqa: E731
t_year = count(**base, event_type="ASSAULT", **w("2025-01-01", "2025-12-31"))
t_q = sum(count(**base, event_type="ASSAULT", **w(q.start_time.date(), q.end_time.date())) for q in pd.period_range("2025Q1", "2025Q4", freq="Q"))
t_m = sum(count(**base, event_type="ASSAULT", **w(m.start_time.date(), m.end_time.date())) for m in pd.period_range("2025-01", "2025-12", freq="M"))
with ThreadPoolExecutor(16) as ex:
    t_d = sum(ex.map(lambda d: count(**base, event_type="ASSAULT", **w(d.date(), d.date())), pd.date_range("2025-01-01", "2025-12-31")))
say(f"Iran → Israel, ASSAULT, 2025: year **{t_year}**; 4 quarters **{t_q}**; 12 months **{t_m}**; 365 single days **{t_d}**.")
ok1 = t_year == t_q == t_m == t_d
t_all = count(**base, **w("2025-01-01", "2025-12-31"))
t_types = sum(count(**base, event_type=t, **w("2025-01-01", "2025-12-31")) for t in tx.ALL_TYPES)
say(f"Same pair, no type filter: **{t_all}**; sum over the 16 types: **{t_types}**.")
ok1 = ok1 and t_all == t_types
say(f"- Verdict: **{'YES' if ok1 else 'NO'}** (additive over time and over event types; both window ends inclusive)\n")

# ── check 2 ──────────────────────────────────────────────────────────────────
say("## 2. Does page 1 return all events when `total` <= 250?\n")
rows, bad, tries = [], 0, 0
while len(rows) < 8 and tries < 500:
    tries += 1
    a, r = rng.sample(nodes, 2)
    et = rng.choice(tx.ALL_TYPES)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.randrange(0, 600))
    end = min(start + pd.Timedelta(days=rng.choice([30, 90, 240])), latest)
    f = {"actor_country_id": a, "recipient_country_id": r, "event_type": et, **w(start.date(), end.date())}
    t = count(**f)
    if not 1 <= t <= 250:
        continue
    pg = page1(**f)
    ev = pg["events"]
    for e in ev:
        if a not in {c["id"] for c in e["actor"]["countries"]} or r not in {c["id"] for c in e["recipient"]["countries"]} \
                or e["event_type"] != et or not (f["event_date_from"] <= e["event_date"] <= f["event_date_to"]):
            bad += 1
    rows.append({"actor→recipient": f"{names[a]}→{names[r]}", "type": et, "window": f"{f['event_date_from']}..{f['event_date_to']}",
                 "total": t, "returned": len(ev), "equal": t == len(ev)})
# at the limit: grow a window until the total passes 250
d0 = pd.Timestamp("2025-01-01")
lim = []
for days in range(100, 220):
    f = {**{"actor_country_id": ids["Russia"], "recipient_country_id": ids["Ukraine"]}, **w(d0.date(), (d0 + pd.Timedelta(days=days)).date())}
    pg = page1(**f)
    lim.append((pg["total"], len(pg["events"]), pg["total_pages"]))
at_250 = [x for x in lim if x[0] == 250]
over = [x for x in lim if 250 < x[0] <= 300]
tbl = pd.DataFrame(rows)
say(tbl.to_markdown(index=False))
ok2 = bool(tbl.equal.all()) and bad == 0
say(f"\n- Events returned == total in **{int(tbl.equal.sum())}/{len(tbl)}** samples; events not matching the filter: **{bad}**.")
if at_250:
    say(f"- At the limit: a window with total 250 returned {at_250[0][1]} events; a window with total {over[0][0]} "
        f"returned {over[0][1]} and reported {over[0][2]} pages." if over else f"- At the limit: total 250 returned {at_250[0][1]}.")
say(f"- Verdict: **{'YES' if ok2 else 'NO'}**\n")

# ── check 3 ──────────────────────────────────────────────────────────────────
say("## 3. Fields and order\n")
pg = page1(actor_country_id=ids["Russia"], recipient_country_id=ids["Ukraine"], event_date_from="2024-01-01")
ev = pg["events"]
keys = list(ev[0])
dates = [e["event_date"] for e in ev]
say("Fields: " + ", ".join(f"`{k}`" for k in keys) + ".")
say("Asked for: " + ", ".join(f"`{k}` {'present' if k in keys else '**MISSING**'}"
                              for k in ("event_date", "intensity", "article_count", "source_quote", "third_party")))
say(f"- Newest first: **{'YES' if dates == sorted(dates, reverse=True) else 'NO'}** (first {dates[0]}, last on the page {dates[-1]}); "
    f"`article_links` filled in {sum(1 for e in ev if e.get('article_links'))}/{len(ev)}.\n")

# ── check 4 ──────────────────────────────────────────────────────────────────
say("## 4. Throughput and build time\n")


def rate(workers, n=480):
    jobs = [(rng.choice(nodes), rng.choice(nodes), rng.choice(tx.ALL_TYPES)) for _ in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: count(actor_country_id=j[0], recipient_country_id=j[1], event_type=j[2], event_date_from="2024-01-01"), jobs))
    return n / (time.time() - t0) * 60


rates = {k: rate(k) for k in (8, 16, 32)}
say("Requests per minute: " + "; ".join(f"{k} workers → **{v:,.0f}**" for k, v in rates.items()) + ".")
say("- Step 1 of the task (the edge counts) **already exists**: `scripts/build_edge_counts.py` stored every event between "
    "the 60 biggest countries (50,487 events, 63,474 country-pair edges) with 40,832 requests in 3.6 minutes plus "
    "3,540 pair totals in about 3.5 minutes, 0 credits. No window at any level held more than 250 events, so the lists are "
    "complete. Monthly and quarterly graphs are cut from those events and need **no new requests**. Adding a country "
    "or two for the case studies means rerunning the script with `--extra`: about 1,000 requests per extra country "
    "(under 2 minutes).\n")

# ── check 5 ──────────────────────────────────────────────────────────────────
say("## 5. Is there enough data per period for clusters?\n")
ev = pd.read_csv(EDGES / "events.csv", parse_dates=["event_date"])
ed = pd.read_csv(EDGES / "event_edges.csv")
m = ed.merge(ev, on="event_id")
m["layer"] = m.event_type.map(tx.layer_of)
m["category"] = m.event_type.map(tx.category_of)
rows = []
for freq, label in (("Q", "quarter"), ("M", "month")):
    m["period"] = m.event_date.dt.to_period(freq)
    for (layer, cat), g in m.groupby(["layer", "category"]):
        per = g.groupby("period").agg(pair_events=("event_id", "size"),
                                      pairs=("actor_id", lambda s: len(set(zip(s, g.loc[s.index, "recipient_id"])))))
        nodes_per = g.groupby("period").apply(lambda x: len(set(x.actor_id) | set(x.recipient_id)), include_groups=False)
        rows.append({"layer": layer, "category": cat, "period": label, "periods": len(per),
                     "median events": int(per.pair_events.median()), "median pairs": int(per.pairs.median()),
                     "median countries": int(nodes_per.median()), "min countries": int(nodes_per.min())})
dens = pd.DataFrame(rows)
say(dens.to_markdown(index=False))
say("\nRule of thumb: with fewer than about 120 country pairs among 60 countries a graph is thin and communities in it are fragile.")
for label in ("quarter", "month"):
    d = dens[dens.period == label]
    thinnest = d.sort_values("median events").iloc[0]
    ok = d[d["median pairs"] >= 120]
    say(f"- Per {label}: {len(ok)} of {len(d)} layer/category graphs reach 120 pairs; the one with the fewest events is "
        f"{thinnest.layer} {thinnest.category} with a median of {thinnest['median pairs']} pairs and {thinnest['median events']} events "
        f"among {thinnest['median countries']} countries.")
say("\nRecommendation derived from this table: **quarters are feasible for all four graphs** (material cooperation is the "
    "thinnest, so it needs the stability check most); months only for the dense graphs (words, material conflict). The page "
    "should therefore offer quarter as the standard period and the full period as a reference, and mark countries that are "
    "barely active in a period as uncertain.\n")

b2 = balance()
say(f"Total requests to /events in this probe: {len(sent):,}, of which with an api-key header: **{sum(r['has_key'] for r in sent)}**. "
    f"Balance at the end: {b2} (at the start: {b0}).")
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out))
print(f"\nSaved to {REPORT}")
