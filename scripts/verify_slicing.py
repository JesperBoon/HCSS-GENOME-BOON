"""
verify_slicing.py: do many small free queries return exactly the same events as the paid paging of one big query?

Our whole data build relies on one claim: a window with at most 250 events is complete on the free first page, so
cutting a query into such windows loses nothing. This script tests that claim against the real thing, once:

  A  one big query (500-750 events), fetched the way a paying user would: page 1 (free, no key) plus pages 2 and 3
     with the API key. Page 1 is deliberately fetched WITHOUT the key, so only pages 2-3 cost credits (about
     total - 250 credits, at most ~500).
  B  the same query cut into windows of at most 250 events (adaptive_windows), each fetched from the free first page.

Both sets of events must be identical, event by event. The script compares ids and the main fields.

Default is a DRY RUN: it finds the query with free counts and prints what it would cost. Add --spend to run A and B.
A hard cap on credits is set from --budget (default 600) and checked before every paid request.

Run (project root):  .venv/bin/python scripts/verify_slicing.py            # plan only, free
                     .venv/bin/python scripts/verify_slicing.py --spend    # really spends the credits
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from edge_counts import Window, adaptive_windows            # noqa: E402
from genome_api import GenomeClient, PAGE_SIZE               # noqa: E402

REPORT = ROOT / "data" / "processed" / "slicing_verification.json"
CANDIDATES = [("Russia", "Ukraine", "ASSAULT"), ("Ukraine", "Russia", "ASSAULT"), ("Israel", "Palestine", "ASSAULT"),
              ("Russia", "Ukraine", "ACCUSE"), ("Iran", "Israel", "ACCUSE"), ("USA", "Iran", "ACCUSE"),
              ("USA", "Ukraine", None), ("Israel", "Iran", None), ("Russia", "USA", None), ("China", "USA", None)]
FIELDS = ("id", "event_date", "event_type", "category", "intensity", "article_count", "core_sentence")


def key(e):
    return tuple(json.dumps(e[f], sort_keys=True) for f in FIELDS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spend", action="store_true", help="really use credits (default: plan only)")
    ap.add_argument("--budget", type=int, default=600, help="hard cap on credits for this run")
    ap.add_argument("--since", default="2024-01-01")
    args = ap.parse_args()

    free = GenomeClient(free_only=True)
    ids = {c["name"]: c["id"] for c in free.countries()["countries"]}
    latest = date.fromisoformat(free._get("/events/time-range")["latest"])

    def total_until(base, end):
        return free.count(**{**base, "event_date_to": str(end)})

    best = None
    for a, r, et in CANDIDATES:
        f = {"actor_country_id": ids[a], "recipient_country_id": ids[r], "event_date_from": args.since,
             "event_date_to": str(latest), **({"event_type": et} if et else {})}
        total = free.count(**f)
        end = latest
        if total > 740:                                       # shorten the period until the total is about 700
            lo, hi = date.fromisoformat(args.since), latest
            while (hi - lo).days > 1:
                mid = lo + (hi - lo) / 2
                if total_until(f, mid) > 700:
                    hi = mid
                else:
                    lo = mid
            end, total = lo, total_until(f, lo)
            f = {**f, "event_date_to": str(end)}
        print(f"  {a}->{r} {et or 'any type'} until {end}: total {total}")
        if 500 <= total <= 740 and (best is None or total > best[0]):
            best = (total, f, f"{a} → {r} ({et or 'any type'})", end)
    if best is None:
        sys.exit("No candidate query with 500-740 events; add one to CANDIDATES.")
    total, f, label, latest = best
    paid_pages = list(range(2, (total + PAGE_SIZE - 1) // PAGE_SIZE + 1))
    cost = total - PAGE_SIZE
    print(f"\nChosen query: {label}, {args.since} to {latest}: {total} events, {len(paid_pages) + 1} pages.")
    print(f"Page 1 is fetched without the key (free). Pages {paid_pages} cost about {cost} credits (cap {args.budget}).")
    if cost > args.budget:
        sys.exit("Cost above the cap; aborting.")
    if not args.spend:
        print("\nDry run: nothing spent. Add --spend to run the comparison.")
        return

    paid = GenomeClient()
    paid.budget = args.budget
    before = paid.balance()
    print(f"Balance before: used {before['used_credits']}, left {before['remaining_credits']}")
    # A: paging
    pages = [free.events_page(1, cache=False, **f)["events"]]
    for p in paid_pages:
        pages.append(paid.events_page(p, cache=False, **f)["events"])
    A = [e for pg in pages for e in pg]
    after = paid.balance()
    spent = after["used_credits"] - before["used_credits"]
    print(f"Paging returned {len(A)} events; credits spent {spent} (balance now {after['remaining_credits']}).")

    # B: free slicing
    whole = Window(date.fromisoformat(args.since), latest)
    leaves = adaptive_windows(lambda w: free.count(**{**f, "event_date_from": str(w.start), "event_date_to": str(w.end)}),
                              whole, "week", known_total=total)
    B = []
    for leaf in leaves:
        pg = free.events_page(1, cache=False, **{**f, "event_date_from": str(leaf.window.start), "event_date_to": str(leaf.window.end)})
        assert len(pg["events"]) == leaf.total, f"window {leaf.window} not complete"
        B.extend(pg["events"])
    ka, kb = {key(e) for e in A}, {key(e) for e in B}
    ids_a, ids_b = {e["id"] for e in A}, {e["id"] for e in B}
    ok = ids_a == ids_b and ka == kb and len(A) == len(B) == total and len(ids_a) == total
    result = {"query": label, "since": args.since, "until": str(latest), "total": total, "paged_events": len(A), "sliced_events": len(B),
              "slices": len(leaves), "ids_equal": ids_a == ids_b, "fields_equal": ka == kb, "duplicates_in_paging": len(A) - len(ids_a),
              "credits_spent": spent, "balance_before": before["remaining_credits"], "balance_after": after["remaining_credits"],
              "identical": ok}
    REPORT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
