"""
build_event_text.py: enrich the stored events with their text and context. FREE, no credits.

The edge build (build_edge_counts.py) keeps only the small fields of each event. Here every closed window is fetched
once more from the free first page, this time keeping what is needed to understand and clean the events:
the summary, the verbatim quote, the core sentence, the names (not only the countries) of actor, recipient and third
party, the locations, and the first article links.

Input : data/processed/edges/leaf_counts.csv (each row is one window that holds at most 250 events)
Output: data/processed/events_text.parquet   (one row per event id)

Uses a free-only client: any request that needs the API key raises before it is sent.
Run (project root):  .venv/bin/python scripts/build_event_text.py            (about 2-4 minutes)
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from genome_api import GenomeClient   # noqa: E402

OUT = ROOT / "data" / "processed" / "events_text.parquet"
client = GenomeClient(free_only=True)


def names(role):
    return "; ".join(role.get("normalized_names") or role.get("raw_names") or [])


def row(e):
    return {"event_id": e["id"], "event_summary": e["event_summary"], "source_quote": e["source_quote"],
            "core_sentence": e["core_sentence"], "actor_names": names(e["actor"]), "recipient_names": names(e["recipient"]),
            "third_party_names": names(e["third_party"]),
            "third_party_countries": "; ".join(c["name"] for c in e["third_party"]["countries"]),
            "location_names": names(e["location"]), "location_countries": "; ".join(c["name"] for c in e["location"]["countries"]),
            "article_links": " ".join((e.get("article_links") or [])[:3])}


def fetch(w):
    f = {"actor_country_id": int(w.actor_id), "recipient_country_id": int(w.recipient_id), "event_type": w.event_type,
         "event_date_from": w.window_start, "event_date_to": w.window_end}
    for attempt in range(4):
        try:
            page = client.events_page(1, cache=False, **f)
            if len(page["events"]) != int(w.count):
                raise RuntimeError(f"{f}: page 1 has {len(page['events'])} events, expected {w.count}")
            return [row(e) for e in page["events"]]
        except RuntimeError:
            raise
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed after retries: {f}")


def main():
    leaves = pd.read_csv(ROOT / "data/processed/edges/leaf_counts.csv")
    leaves = leaves[leaves.source == "events"].drop_duplicates(["actor_id", "recipient_id", "event_type", "window_start", "window_end"])
    print(f"{len(leaves):,} windows to fetch", flush=True)
    rows, t0 = {}, time.time()
    with ThreadPoolExecutor(16) as ex:
        for i, batch in enumerate(ex.map(fetch, list(leaves.itertuples())), 1):
            for r in batch:
                rows[r["event_id"]] = r
            if i % 1000 == 0:
                print(f"  {i:,}/{len(leaves):,} windows, {len(rows):,} events, {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows.values())
    stored = pd.read_csv(ROOT / "data/processed/edges/events.csv", usecols=["event_id"])
    missing = set(stored.event_id) - set(df.event_id)
    print(f"{len(df):,} events with text; stored events without text: {len(missing)}")
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
