"""
probe_chains.py — step 0 of the "support chains" work: what can the FREE API really do?

FREE: only unauthenticated page-1 requests; no key is read and no credits can be spent.
Report: printed, and saved to data/cache/probe_chains_report.md.

Checks
  1  Is `total` exact for actor_country_id + recipient_country_id + event_type + date window?
     (additivity over time windows and over event types; are both window ends inclusive?)
  2  Does page 1 return ALL events when total <= 250? (compared with total, on random samples)
  3  Which fields does an event have, and is the order newest-first?
  4  Throughput (requests per minute) and an estimate of the cost of building the edge counts.

Run (project root):  .venv/bin/python scripts/probe_chains.py
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

B = "https://genome.app.hcss.nl/api"
ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "cache" / "probe_chains_report.md"
MATERIAL = ["AID", "COOPERATE", "RETREAT", "ASSAULT", "SANCTION", "COERCE", "MOBILIZE", "PROTEST"]
VERBAL = ["AGREE", "CONSULT", "SUPPORT", "CONCEDE", "REQUEST", "ACCUSE", "REJECT", "THREATEN"]
START = "2024-01-01"
TOP_N = 40

rng = random.Random(42)
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64))
out: list[str] = []
n_requests = 0


def say(text=""):
    print(text, flush=True)
    out.append(text)


def page1(**params):
    """One free page-1 request. Retries a few times; counts requests."""
    global n_requests
    for attempt in range(4):
        try:
            n_requests += 1
            r = session.get(f"{B}/events", params={**params, "page": 1}, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"request failed: {params}")


def total(**params):
    return page1(**params)["total"]


def window(a, b):
    return {"event_date_from": str(a), "event_date_to": str(b)}


# ── set-up: the countries ────────────────────────────────────────────────────
ids = {c["name"]: c["id"] for c in session.get(f"{B}/countries").json()["countries"]}
name_of = {v: k for k, v in ids.items()}
monthly = pd.read_csv(ROOT / "data/processed/monthly_counts.csv")
top = monthly.groupby("country_id").n.sum().sort_values(ascending=False).head(TOP_N).index.tolist()
latest = pd.Timestamp(session.get(f"{B}/events/time-range").json()["latest"])
say("# Probe: can free count queries carry the support-chain analysis?\n")
say(f"Run on {pd.Timestamp.now():%Y-%m-%d %H:%M}; data runs until {latest.date()}; top-{TOP_N} countries by volume.\n")
say("Country set: " + ", ".join(name_of[c] for c in top) + ".\n")

RU, UA = ids["Russia"], ids["Ukraine"]
base = {"actor_country_id": RU, "recipient_country_id": UA}

# ── check 1: exactness of `total` ────────────────────────────────────────────
say("## 1. Is `total` exact?\n")
year = pd.date_range("2025-01-01", "2025-12-31", freq="D")
t_year = total(**base, event_type="ASSAULT", **window("2025-01-01", "2025-12-31"))
quarters = pd.period_range("2025Q1", "2025Q4", freq="Q")
months = pd.period_range("2025-01", "2025-12", freq="M")
t_q = sum(total(**base, event_type="ASSAULT", **window(q.start_time.date(), q.end_time.date())) for q in quarters)
t_m = sum(total(**base, event_type="ASSAULT", **window(m.start_time.date(), m.end_time.date())) for m in months)
with ThreadPoolExecutor(16) as ex:
    t_d = sum(ex.map(lambda d: total(**base, event_type="ASSAULT", **window(d.date(), d.date())), year))
say(f"Russia → Ukraine, ASSAULT, calendar year 2025: whole year **{t_year}**; sum of 4 quarters **{t_q}**; "
    f"sum of 12 months **{t_m}**; sum of 365 single days **{t_d}**.")
ok_time = t_year == t_q == t_m == t_d
say(f"- Additive over time windows, and both window ends are inclusive (single-day windows add up): "
    f"**{'YES' if ok_time else 'NO'}**\n")

types16 = MATERIAL + VERBAL
t_all = total(**base, **window("2025-01-01", "2025-12-31"))
t_cat = sum(total(**base, category=c, **window("2025-01-01", "2025-12-31")) for c in ("COOPERATION", "CONFLICT"))
t_types = sum(total(**base, event_type=t, **window("2025-01-01", "2025-12-31")) for t in types16)
say(f"Same pair and year, no type filter: **{t_all}**; COOPERATION + CONFLICT: **{t_cat}**; "
    f"sum over the 16 event types: **{t_types}**.")
ok_types = t_all == t_cat == t_types
say(f"- Additive over event types and categories: **{'YES' if ok_types else 'NO'}**\n")

# ── check 2: does page 1 hold everything when total <= 250? ──────────────────
say("## 2. Does page 1 return all events when `total` <= 250?\n")
samples, tries = [], 0
while len(samples) < 12 and tries < 600:
    tries += 1
    a, r = rng.sample(top, 2)
    et = rng.choice(MATERIAL)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.randrange(0, 600))
    end = start + pd.Timedelta(days=rng.choice([30, 60, 120, 240]))
    end = min(end, latest)
    p = {"actor_country_id": a, "recipient_country_id": r, "event_type": et, **window(start.date(), end.date())}
    d = page1(**p)
    if 1 <= d["total"] <= 250:
        samples.append((p, d))
rows, filter_violations, type_violations, date_violations, dup = [], 0, 0, 0, 0
for p, d in samples:
    ev = d["events"]
    ids_ = [e["id"] for e in ev]
    dup += len(ids_) - len(set(ids_))
    for e in ev:
        if p["actor_country_id"] not in {c["id"] for c in e["actor"]["countries"]}:
            filter_violations += 1
        if p["recipient_country_id"] not in {c["id"] for c in e["recipient"]["countries"]}:
            filter_violations += 1
        if e["event_type"] != p["event_type"]:
            type_violations += 1
        if not (p["event_date_from"] <= e["event_date"] <= p["event_date_to"]):
            date_violations += 1
    rows.append({"actor→recipient": f"{name_of[p['actor_country_id']]}→{name_of[p['recipient_country_id']]}",
                 "type": p["event_type"], "window": f"{p['event_date_from']}..{p['event_date_to']}",
                 "total": d["total"], "events returned": len(ev), "total_pages": d["total_pages"],
                 "equal": d["total"] == len(ev)})
tbl = pd.DataFrame(rows)
say(tbl.to_markdown(index=False))
all_equal = bool(tbl.equal.all())
say(f"\n- Events returned == total in **{int(tbl.equal.sum())}/{len(tbl)}** samples "
    f"({tries} random pair/window draws were needed to find them).")
say(f"- Duplicate ids on a page: **{dup}**; events whose actor/recipient country does not match the filter: "
    f"**{filter_violations}**; wrong event type: **{type_violations}**; date outside the window: **{date_violations}**.")
say(f"- Verdict: **{'YES' if all_equal and not (filter_violations or type_violations or date_violations or dup) else 'NO'}**\n")

# ── check 3: fields and order ────────────────────────────────────────────────
say("## 3. Fields and order\n")
big = page1(**base, **window(START, str(latest.date())))
ev = big["events"]
say(f"Russia → Ukraine since {START}: total **{big['total']}**, page size {big['page_size']}, "
    f"total_pages {big['total_pages']}, returned {len(ev)}.\n")
keys = list(ev[0].keys())
say("Event fields: " + ", ".join(f"`{k}`" for k in keys) + "\n")
wanted = ["event_date", "intensity", "article_count", "source_quote", "third_party"]
say("Fields asked for: " + ", ".join(f"`{k}` {'present' if k in keys else '**MISSING**'}" for k in wanted) + "\n")
dates = [e["event_date"] for e in ev]
newest_first = all(dates[i] >= dates[i + 1] for i in range(len(dates) - 1))
say(f"- Newest first on page 1: **{'YES' if newest_first else 'NO'}** "
    f"(first {dates[0]}, last on the page {dates[-1]}).")
nonempty_tp = sum(1 for e in ev if e["third_party"]["normalized_names"])
say(f"- `third_party` is filled in {nonempty_tp}/{len(ev)} of these events; `article_count` ranges "
    f"{min(e['article_count'] for e in ev)}–{max(e['article_count'] for e in ev)}; intensity "
    f"{min(e['intensity'] for e in ev if e['intensity'] is not None):.2f} to "
    f"{max(e['intensity'] for e in ev if e['intensity'] is not None):.2f}.")
links_filled = sum(1 for e in ev if e.get("article_links"))
say(f"- `article_links` (source URLs) is filled in {links_filled}/{len(ev)} events on the free page 1; "
    f"`event_summary` and `source_quote` are filled in {sum(1 for e in ev if e.get('event_summary'))} and "
    f"{sum(1 for e in ev if e.get('source_quote'))}.")
ex = ev[0]
say("\nOne event (trimmed):\n")
say("```\n" + "\n".join([
    f"id            {ex['id']}", f"event_date    {ex['event_date']}", f"event_type    {ex['event_type']}",
    f"intensity     {ex['intensity']}", f"article_count {ex['article_count']}",
    f"core_sentence {ex['core_sentence']}", f"actor         {ex['actor']['normalized_names']} {[c['name'] for c in ex['actor']['countries']]}",
    f"recipient     {ex['recipient']['normalized_names']} {[c['name'] for c in ex['recipient']['countries']]}",
    f"third_party   {ex['third_party']['normalized_names']}", f"source_quote  {ex['source_quote'][:110]}…"]) + "\n```\n")

# ── check 4: throughput and cost estimate ────────────────────────────────────
say("## 4. Throughput and cost of the edge counts\n")


def rate(workers, n=480):
    pairs = [(rng.choice(top), rng.choice(top)) for _ in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex_:
        list(ex_.map(lambda ar: total(actor_country_id=ar[0], recipient_country_id=ar[1],
                                      event_type=rng.choice(MATERIAL), event_date_from=START), pairs))
    return n / (time.time() - t0) * 60


rates = {w: rate(w) for w in (8, 16, 32)}
best = max(rates.values())
say("Requests per minute: " + "; ".join(f"{w} workers → **{v:,.0f}**" for w, v in rates.items()) + ".\n")

# adaptive splitting, simulated on a random sample of (pair, type) cells
quarters_all = pd.period_range("2024Q1", latest.strftime("%Y-%m"), freq="Q")


def month_windows(q):
    return [m for m in pd.period_range(q.start_time, q.end_time, freq="M")]


def week_windows(m):
    start, end = m.start_time.normalize(), m.end_time.normalize()
    cuts = [start] + [d for d in pd.date_range(start, end) if d.dayofweek == 0 and d != start]
    return [(c, (cuts[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(cuts) else end) for i, c in enumerate(cuts)]


def cell_cost(a, r, et, finest):
    """Requests needed for one (actor, recipient, type) cell with the adaptive scheme; finest in {quarter, month, week}."""
    req, ev_pages, ev_rows = 1, 0, 0
    t = total(actor_country_id=a, recipient_country_id=r, event_type=et, event_date_from=START)
    if t == 0:
        return req, 0, 0, 0
    if t <= 250:
        return req + 1, 1, t, t
    levels = ["quarter", "month", "week"][: ["quarter", "month", "week"].index(finest) + 1]
    for q in quarters_all:
        req += 1
        tq = total(actor_country_id=a, recipient_country_id=r, event_type=et,
                   **window(q.start_time.date(), min(q.end_time.date(), latest.date())))
        if tq == 0:
            continue
        if tq <= 250:
            req += 1; ev_pages += 1; ev_rows += tq
            continue
        if "month" not in levels:
            continue
        for m in month_windows(q):
            req += 1
            tm = total(actor_country_id=a, recipient_country_id=r, event_type=et,
                       **window(m.start_time.date(), min(m.end_time.date(), latest.date())))
            if tm == 0:
                continue
            if tm <= 250:
                req += 1; ev_pages += 1; ev_rows += tm
                continue
            if "week" not in levels:
                continue
            for w0, w1 in week_windows(m):
                req += 1
                tw = total(actor_country_id=a, recipient_country_id=r, event_type=et,
                           **window(w0.date(), min(w1.date(), latest.date())))
                if 0 < tw <= 250:
                    req += 1; ev_pages += 1; ev_rows += tw
    return req, ev_pages, ev_rows, t


SAMPLE_PAIRS = [tuple(rng.sample(top, 2)) for _ in range(150)]   # the same pairs for every layer


def estimate(types, finest):
    """Scale a per-pair sample to all ordered pairs; return values with standard errors."""
    def one_pair(pair):
        res = [cell_cost(pair[0], pair[1], et, finest) for et in types]
        return (sum(x[0] for x in res), sum(x[1] for x in res), sum(x[2] for x in res),
                sum(1 for x in res if x[3] > 0), len(res))
    with ThreadPoolExecutor(16) as ex_:
        per_pair = pd.DataFrame(list(ex_.map(one_pair, SAMPLE_PAIRS)),
                                columns=["reqs", "pages", "rows", "nonzero", "cells"])
    n_all = TOP_N * (TOP_N - 1)
    def scaled(col):
        m, se = per_pair[col].mean() * n_all, per_pair[col].std() / len(per_pair) ** 0.5 * n_all
        return m, se
    return n_all * len(types), per_pair.nonzero.sum() / per_pair.cells.sum(), scaled("reqs"), scaled("pages"), scaled("rows")


say(f"Estimated from the same random sample of 150 country pairs × the event types of each layer, scaled to all "
    f"{TOP_N}×{TOP_N - 1} ordered pairs (adaptive splitting: whole period → quarter → month → week, stopping as "
    f"soon as a window has ≤ 250 events, which are then taken from page 1):\n")
est = []
for layer, types, finest in [("Material layer, down to weeks", MATERIAL, "week"),
                             ("Material layer, down to months", MATERIAL, "month"),
                             ("Word layer, down to months", VERBAL, "month")]:
    full, nz, (reqs, reqs_se), (pages, pages_se), (rows_, rows_se) = estimate(types, finest)
    est.append({"layer": layer, "cells (pair×type)": f"{full:,}", "share with events": f"{nz:.0%}",
                "requests": f"{reqs:,.0f} ± {reqs_se:,.0f}",
                f"minutes at {best:,.0f}/min": f"{reqs / best:,.1f}",
                "page-1 event windows": f"{pages:,.0f} ± {pages_se:,.0f}",
                "events stored": f"{rows_:,.0f} ± {rows_se:,.0f}"})
say(pd.DataFrame(est).to_markdown(index=False))
say("\n(± is one standard error of the sampling. A few busy pairs dominate the number of events, so the "
    "events estimate is the least precise one.)")
say(f"\nTotal requests made by this probe: {n_requests:,}. Credits spent: **0** (no key is ever sent).")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(out))
print(f"\nSaved to {REPORT}")
