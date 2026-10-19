"""
check_api.py — verify the API connection and show the credit balance.

Run from the project root:  .venv/bin/python scripts/check_api.py
Costs no credits (uses only free endpoints and /keys/me).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genome_api import GenomeClient, api_key


def main():
    c = GenomeClient()
    s = c.summary()["totals"]
    print(f"Public API reachable: {s['total_events']:,} events, {s['earliest_event_date']} to {s['latest_event_date']}")

    if not api_key():
        print("No valid key in .env yet. Paste your genome_... key into .env and rerun.")
        return
    info = c.balance()
    print("API key accepted. /keys/me returned:")
    for k, v in info.items():
        # never print the key itself
        if "key" in k.lower() and isinstance(v, str) and v.startswith("genome_"):
            v = v[:10] + "…"
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
