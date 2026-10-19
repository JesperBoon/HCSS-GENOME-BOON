"""
genome_api.py

Thin client for the GENOME Events API (https://genome.app.hcss.nl/api).

Credit rules (see API documentation):
    - Page 1 of /events without a key is FREE and always returns the exact `total`
      for the applied filters, so counts cost nothing.
    - Authenticated requests cost 1 credit per event returned.
    - Every response is cached on disk, so an event page is never paid for twice.
    - A per-session budget (GENOME_CREDIT_BUDGET) stops accidental overspending.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://genome.app.hcss.nl/api"
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
PAGE_SIZE = 250
FREE_PAGE_TTL = 12 * 3600  # seconds before a cached free page 1 is refetched

# Filters confirmed to work on /events (others, e.g. `country` and `q`, are ignored).
SUPPORTED_FILTERS = {
    "category", "event_type", "event_date_from", "event_date_to",
    "actor_country", "recipient_country", "third_party_country", "location_country",
    "actor_country_id", "recipient_country_id", "intensity_min",
}


class BudgetExceeded(RuntimeError):
    pass


class PaidRequestBlocked(RuntimeError):
    """Raised in free-only mode when code tries to use the API key (page 2 and beyond, details)."""


def _load_env() -> None:
    """Minimal .env reader (avoids an extra dependency). Real env vars win."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()


def api_key() -> str | None:
    key = os.environ.get("GENOME_API_KEY", "")
    return key if key.startswith("genome_") and "your_key_here" not in key else None


class GenomeClient:
    def __init__(self, free_only: bool = False):
        """free_only=True makes any request that needs the API key fail before it is sent,
        so a bug can never spend credits. Counts and page 1 of /events stay available."""
        self.free_only = free_only
        self.session = requests.Session()
        self.budget = int(os.environ.get("GENOME_CREDIT_BUDGET", "2000"))
        self.spent = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── low level ──────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None, auth: bool = False):
        headers = {}
        if auth and self.free_only:
            raise PaidRequestBlocked(f"{path} needs the API key; this client is free-only")
        if auth:
            key = api_key()
            if not key:
                raise RuntimeError("No valid GENOME_API_KEY found in .env")
            headers["x-api-key"] = key
        r = self.session.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=60)
        if r.status_code == 402:
            raise RuntimeError(f"Out of credits: {r.json().get('detail')}")
        r.raise_for_status()
        return r.json()

    # ── free reference data ────────────────────────────────────────────────
    def summary(self) -> dict:
        return self._get("/stats/summary")

    def countries(self):
        return self._get("/countries")

    def count(self, **filters) -> int:
        """Exact number of events matching the filters. Free (unauthenticated page 1)."""
        self._check_filters(filters)
        return self._get("/events", {**filters, "page": 1})["total"]

    # ── account ────────────────────────────────────────────────────────────
    def balance(self) -> dict:
        """Key info and remaining credits. Costs nothing."""
        return self._get("/keys/me", auth=True)

    # ── events (cached, budget-guarded) ────────────────────────────────────
    def events_page(self, page: int = 1, cache: bool = True, **filters) -> dict:
        """One page of events. cache=False skips the disk cache (for bulk jobs that store their own results)."""
        if page > 1 and self.free_only:
            raise PaidRequestBlocked(f"page {page} costs credits; this client is free-only")
        self._check_filters(filters)
        params = {**filters, "page": page}
        cache_file = CACHE_DIR / f"events_{self._hash(params)}.json"
        # Page 1 is "newest first", so it goes stale; paid pages are kept for good.
        if cache and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if page > 1 or age < FREE_PAGE_TTL:
                return json.loads(cache_file.read_text())

        # Page 1 is free without a key; deeper pages need the key and cost credits.
        paid = page > 1
        if paid and self.spent + PAGE_SIZE > self.budget:
            # Checked BEFORE the request: a page can cost up to PAGE_SIZE credits and the credits are gone once it is sent.
            raise BudgetExceeded(
                f"A paid page can cost up to {PAGE_SIZE} credits and only {self.budget - self.spent} of this session's "
                f"budget ({self.budget}) is left; nothing was requested."
            )
        data = self._get("/events", params, auth=paid)
        if paid:
            self.spent += len(data["events"])
        if cache:
            cache_file.write_text(json.dumps(data))
        return data

    def events(self, max_events: int = 250, **filters) -> list[dict]:
        """Fetch up to max_events matching events, page by page."""
        out, page = [], 1
        while len(out) < max_events:
            data = self.events_page(page, **filters)
            out.extend(data["events"])
            if page >= data["total_pages"] or not data["events"]:
                break
            page += 1
        return out[:max_events]

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _hash(params: dict) -> str:
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def _check_filters(filters: dict) -> None:
        bad = set(filters) - SUPPORTED_FILTERS
        if bad:
            raise ValueError(f"Unsupported filter(s) {sorted(bad)}; supported: {sorted(SUPPORTED_FILTERS)}")
