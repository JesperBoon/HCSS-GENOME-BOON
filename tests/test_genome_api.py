"""Tests for src/genome_api.py with a fake HTTP session: no network, no credits."""

import time

import pytest

import genome_api as api


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records every call; returns a page of `n_events` events (or a 402 when `status` says so)."""

    def __init__(self, n_events=250, status=200):
        self.calls, self.n_events, self.status = [], n_events, status

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        if self.status == 402:
            return FakeResponse({"detail": "Insufficient credits, 0 left"}, 402)
        return FakeResponse({"total": 1234, "page": params.get("page", 1), "page_size": 250, "total_pages": 5,
                             "events": [{"id": f"e{i}"} for i in range(self.n_events)]})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("GENOME_API_KEY", "genome_testkey")
    monkeypatch.setenv("GENOME_CREDIT_BUDGET", "300")
    c = api.GenomeClient()
    c.session = FakeSession()
    return c


def test_count_returns_the_total_without_a_key(client, monkeypatch):
    monkeypatch.delenv("GENOME_API_KEY")
    assert client.count(event_type="ASSAULT") == 1234
    call = client.session.calls[0]
    assert "x-api-key" not in call["headers"] and call["params"]["page"] == 1
    assert client.spent == 0


def test_unsupported_filters_are_rejected_because_the_api_silently_ignores_them(client):
    with pytest.raises(ValueError, match="Unsupported filter"):
        client.count(country="Ukraine")


def test_free_first_page_costs_no_credits_and_is_cached(client):
    first = client.events_page(1, actor_country_id=5)
    second = client.events_page(1, actor_country_id=5)
    assert first == second
    assert len(client.session.calls) == 1             # the second call came from the disk cache
    assert "x-api-key" not in client.session.calls[0]["headers"]
    assert client.spent == 0


def test_a_stale_free_page_is_fetched_again(client, monkeypatch):
    client.events_page(1, actor_country_id=5)
    monkeypatch.setattr(api, "FREE_PAGE_TTL", 0)
    time.sleep(0.01)
    client.events_page(1, actor_country_id=5)
    assert len(client.session.calls) == 2


def test_paid_pages_send_the_key_and_count_the_credits(client):
    client.events_page(2, actor_country_id=5)
    assert client.session.calls[0]["headers"]["x-api-key"] == "genome_testkey"
    assert client.spent == 250


def test_paid_pages_are_cached_for_good(client, monkeypatch):
    client.events_page(2, actor_country_id=5)
    monkeypatch.setattr(api, "FREE_PAGE_TTL", 0)
    client.events_page(2, actor_country_id=5)
    assert len(client.session.calls) == 1 and client.spent == 250


def test_the_budget_guard_stops_a_page_that_would_overspend(client):
    client.events_page(2, actor_country_id=5)         # 250 of 300
    with pytest.raises(api.BudgetExceeded):
        client.events_page(3, actor_country_id=5)     # would be 500 of 300
    assert client.spent == 250


def test_the_budget_guard_does_not_cache_a_refused_page(client):
    client.budget = 100
    with pytest.raises(api.BudgetExceeded):
        client.events_page(2, actor_country_id=5)
    assert not list(api.CACHE_DIR.glob("events_*.json"))


def test_a_paid_request_without_a_valid_key_fails_before_calling_the_api(client, monkeypatch):
    monkeypatch.setenv("GENOME_API_KEY", "genome_your_key_here")     # the placeholder from .env.example
    with pytest.raises(RuntimeError, match="GENOME_API_KEY"):
        client.events_page(2, actor_country_id=5)
    assert client.session.calls == []


def test_http_402_is_reported_as_out_of_credits(client):
    client.session = FakeSession(status=402)
    with pytest.raises(RuntimeError, match="Out of credits"):
        client.events_page(2, actor_country_id=5)


def test_events_collects_pages_until_max_events(client):
    out = client.events(max_events=300, actor_country_id=5)
    assert len(out) == 300
    assert client.spent == 250                         # page 1 free, page 2 paid


def test_api_key_helper_rejects_placeholders(monkeypatch):
    monkeypatch.setenv("GENOME_API_KEY", "genome_your_key_here")
    assert api.api_key() is None
    monkeypatch.setenv("GENOME_API_KEY", "not-a-key")
    assert api.api_key() is None
    monkeypatch.setenv("GENOME_API_KEY", "genome_abc123")
    assert api.api_key() == "genome_abc123"


def test_free_only_client_blocks_paid_pages_before_any_request(client):
    client.free_only = True
    with pytest.raises(api.PaidRequestBlocked):
        client.events_page(2, actor_country_id=5)
    assert client.session.calls == [] and client.spent == 0


def test_free_only_client_blocks_a_cached_paid_page_too(client):
    client.events_page(2, actor_country_id=5)         # cached while the client was still allowed to pay
    client.free_only = True
    with pytest.raises(api.PaidRequestBlocked):
        client.events_page(2, actor_country_id=5)


def test_free_only_client_blocks_key_endpoints(client):
    client.free_only = True
    with pytest.raises(api.PaidRequestBlocked):
        client.balance()
    assert client.session.calls == []


def test_free_only_client_still_serves_counts_and_the_free_first_page(client):
    client.free_only = True
    assert client.count(event_type="ASSAULT") == 1234
    assert len(client.events_page(1, actor_country_id=5)["events"]) == 250
    assert all("x-api-key" not in c["headers"] for c in client.session.calls)


def test_cache_false_neither_reads_nor_writes_the_disk_cache(client):
    client.events_page(1, actor_country_id=5)                       # fills the cache
    client.events_page(1, cache=False, actor_country_id=5)          # must go to the network again
    assert len(client.session.calls) == 2
    client.events_page(1, cache=False, actor_country_id=6)
    assert len(list(api.CACHE_DIR.glob("events_*.json"))) == 1      # the uncached call left no file


def test_the_budget_guard_refuses_before_sending_anything(client):
    """The credits are gone as soon as a paid page is requested, so the check has to come first."""
    client.budget = 200                                    # less than one page
    with pytest.raises(api.BudgetExceeded):
        client.events_page(2, actor_country_id=5)
    assert client.session.calls == [] and client.spent == 0


def test_the_budget_guard_allows_a_page_while_a_whole_page_still_fits(client):
    client.budget = 500
    client.events_page(2, actor_country_id=5)              # 0 + 250 <= 500
    client.events_page(3, actor_country_id=5)              # 250 + 250 <= 500
    assert client.spent == 500
    with pytest.raises(api.BudgetExceeded):
        client.events_page(4, actor_country_id=5)
    assert len(client.session.calls) == 2
