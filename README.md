# Reliability in GENOME

## Open the dashboard

**[hcss-genome-boon.streamlit.app](https://hcss-genome-boon-wx7cuphe7nefwemmcfrkb4.streamlit.app/)** — one link, no
setup, nothing to install.

**[Read how the data was processed](DATA_PROCESSING.md)** — the cleaning, filtering and feature-engineering steps
behind every chart, file by file, with a worked example.

**Do countries mean what they say?** This dashboard asks whether countries' words and their material actions point
the same way, country by country, conflict by conflict, and bloc by bloc, for the HCSS Datalab technical assignment
2026: data processing, analysis and communication of results — the first in [DATA_PROCESSING.md](DATA_PROCESSING.md),
the rest in the app itself.

To run it locally instead:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py          # opens http://localhost:8501
```

No API key needed; the committed data in `data/processed/` is enough to run it after a clone. See section 5 below
for the full setup and section 3 for what each page shows.

---

## 1. Angle one: reliability

Reliability is a property of a **relationship between two countries**, not of one country alone. For a conflict
between two parties (for example Russia and Ukraine) the dashboard asks, for every other country:

| Question | Where |
|---|---|
| **Which side does it take?** | Page 1, section C |
| **Does that move?** (weekly rolling windows, lines and two animations) | Page 1, section B |
| **Do words and actions point the same way?** Friendly words with hostile actions, or the reverse | Page 1, sections C and D |

For a country A and a party T, over a period: `stance(A→T) = (cooperation − conflict) / (cooperation + conflict)`,
from −1 to +1. The **side score** is the stance toward the second party minus the stance toward the first (−2…+2); near
0 (|score| < 0.3) means no clear side. A country only gets a score with enough events (default 3) toward **both**
parties: staying out is not the same as neutral. Event types follow Table 4 of the GENOME paper:

| | Cooperation | Conflict |
|---|---|---|
| **Words** | AGREE, CONSULT, SUPPORT, CONCEDE | REQUEST, ACCUSE, REJECT, THREATEN |
| **Actions** | AID, COOPERATE, RETREAT | ASSAULT, SANCTION, COERCE, MOBILIZE, PROTEST |

This grouping lives in one place, `src/taxonomy.py`; every script, metric and page follows it. An optional switch weights
events by the midpoint of the intensity band of their type (routine 0.15, formal 0.55, material/lethal 0.90).

One example finding: the USA's position in Russia–Ukraine drops from about +1.3 (clearly toward Ukraine) in 2024 to
about 0 around April 2025.

---

## 2. Angle two: support chains

### The idea

Material support often does not go straight from A to the party that needs it. The question here is about **chains**:
A gives aid to B, B gives aid to C, and sometimes a chain ends in a material conflict (B attacks D). The material layer
is where the chains are; the verbal layer is a tool to find material ties we do not see directly.

### What this is and what it is not (also stated on the pages)

- GENOME has **no amounts**: no money, no weapons. Intensity is ordinal. We analyse **reported, time-ordered paths of
  events**, not proven flows. We say *support path*, *candidate path*, *latent tie*; never "A financed C".
- **Closeness in time is not causation.**
- **"Hidden" means "not observed in this English-language newswire corpus"**, not secret.
- **Order is only known at the level of time windows.** Events carry a date, not a time of day, and a window is 2–8
  weeks. A → B and B → C in the *same* window have an unknown order; in the *next* window the order is known. Both are
  reported separately.

### Data: everything for free

A first probe (`scripts/probe_chains.py`, report in `data/cache/probe_chains_report.md`) tested whether the free API can
carry the analysis, with real answers:

| Question | Answer |
|---|---|
| Is `total` exact for actor + recipient + event type + date window? | **Yes.** A whole year, 4 quarters, 12 months and 365 single days all give the same count, so both window ends are inclusive and nothing overlaps. Counts are also additive over event types. |
| Does the free page 1 hold *all* events when `total` ≤ 250? | **Yes**, in 12 of 12 random samples and exactly at the limit (a window with total 250 returns 250; 266 returns 250 and reports a second page). No duplicates, no events that do not match the filter. |
| Fields and order? | `event_date`, `intensity`, `article_count`, `source_quote`, `third_party` are present; newest first. The free first page also contains `article_links`, which the documentation reserves for the paid endpoint. |
| Speed? | 5,500–13,000 requests per minute, depending on the number of parallel workers. |

The edge build (`scripts/build_edge_counts.py`) therefore takes the exact count of every (actor, recipient, event type)
cell for the 60 biggest countries and **splits the time window only while it holds more than 250 events** (whole period →
quarter → month → week); a window with at most 250 events is closed and its events are taken from page 1. In practice
**no window at any level exceeded 250 events**, so the result is the *complete* event list of those 60 countries:
50,487 events, 63,474 country-pair edges, 40,832 requests (plus 3,540 to find the pairs with events), about 8 minutes in total, **0 credits**. A check re-counts every
closed window from the stored events (8,367 windows, 0 mismatches), and a known total from earlier (338 Russian assaults
on Ukraine in 2025) comes out identical.

The client used for this (`src/genome_api.py`) can run in **free-only mode**: any request that needs the API key, such as
page 2 of `/events`, raises `PaidRequestBlocked` before it is sent.

### Method

- **Graph.** `counts[a, r, t]`: events of one layer and category from country a to country r in window t
  (`src/chains.py`). Windows are `window_weeks` wide (2, 4 or 8).
- **Chain weight.** A → B in window t and B → C in window t (same) or t + 1 (next): the weight is the **weaker of the two
  links** (conservative), summed over windows; the product is also available. Chains back to the start (A → B → A) are
  removed.
- **Extras.** Pass-on score per country (the smaller of support received and support given, per window), cycles
  A → B → C → A, and "support → conflict" chains (aid A → B followed by an assault B → D).
- **Null model (essential).** Big countries make chains by volume alone. For each chain we compare the observed weight
  with the weight after moving the timing of every country pair's events at random while keeping its total (1,000
  permutations; the chains that look interesting are re-tested with 20,000). Two versions, because the plain one has a
  flaw:
  - **"Volume only":** an edge's events may move to any window.
  - **"Volume and shared busy weeks":** an edge's events may only move to windows of *similar overall activity*. In a
    war week everything is busy at once; two country pairs that are both busy then do not look "coordinated". The
    tolerance is 8% of the number of windows. Plain shuffling gives 197 "significant" aid-then-attack chains at 2-week
    windows; this version leaves 8.
- **Many tests at once.** Over a thousand chains are tested, so by luck alone dozens look special. We report how many
  chance would give and control the **false discovery rate** (Benjamini–Hochberg, 10%).
- **Words → material support.** `src/hidden.py`: pairs that had at least *k* cooperative-word events in the first two
  thirds of the period (the "treated") are compared with the **most similar** pair without such words, matched on the
  actor's other events, the recipient's other events and the pair's own other events (1:1 nearest neighbour). The
  outcome is a material link in the held-out last third. Confidence intervals bootstrap over actor countries (pairs of
  one actor are not independent). Covariate balance after matching is reported.
- **2 × 2 and candidate latent ties.** Every ordered pair is either *words and material*, *words only*, *material only*
  or *neither*. A chain A → B → C whose end points A and C have words but no observed material tie is a **candidate
  latent tie**: a lead, not a finding.
- **Case studies.** Russia–Ukraine, Iran–Israel and Israel–Palestine, chosen on data facts shown on the page: Ukraine
  receives the most reported aid of all countries (620 events from 32 countries), Palestine is second (373), Russia
  fourth, Iran fifth; Israel is the origin of by far the most assaults; the heaviest aid chain runs USA → Israel →
  Palestine. A Sankey diagram (suppliers → intermediaries → party) with a time slider, and the real events behind each
  path (date, type, intensity, quote, link) from the free first page.
- **Optional narrative.** A short summary written by Claude from the events on screen only (`src/narrative.py`), behind
  a switch and an `ANTHROPIC_API_KEY`; the prompt forbids amounts, causation and outside knowledge and treats quotes as
  data. The dashboard works without it.

### What the data says (as of 18 October 2026)

- **Most apparent support chains are volume.** For aid only (3,161 aid events, 60 countries), 1,168 chains were tested
  at 4-week windows. 49 pass an uncorrected 1% test with the plain comparison (30 with the stricter one) where chance gives about 12;
  **after correction none stands out**, with either version of the null model. With 2-week windows, 16 stand out against volume alone and none once shared
  busy weeks are taken into account. Pass-on scores of the biggest donors (USA, Russia, Israel, India) equal what volume
  predicts. Of 293 loops (A → B → C → A), none beats chance after correction.
- **Aid followed by an attack** (aid A → B, assault B → D): 8 chains survive at 2-week windows, none at 4 or 8 weeks. They
  cluster around Iran as a hub. That they do not hold across window lengths is a reason to treat them as leads.
- **Words do come before new aid, modestly.** Country pairs with at least 3 cooperative statements or talks in the
  first two thirds of the period received a *new* aid link in the last third in 31% of cases, against 17% for
  comparable pairs: **+14.5 points, 95% interval +5 to +23** (173 pairs; balance after matching 0.17). It is not stronger
  with more words, and for three or more new events the interval includes zero. Settings that include existing aid links
  cannot be compared fairly (pairs with words are far more active than any pair without; balance 0.8–0.9) and the
  dashboard says so.
- 485 ordered pairs have words and aid, 302 aid without words, **258 words without aid** (empty words, or support that
  English news does not report), 2,495 neither.

An honest weak result is a result: with free data and a null model that respects busy weeks, the case that *specific*
aid chains are more than coincidence is thin.

### Limitations specific to this angle

- English newswire from one provider; underreported regions are underrepresented.
- **Organisations (EU, NATO, UN) have no country id** in GENOME (`id` is empty in the events), cannot be filtered exactly
  and are not in the graph, although much aid is organised through them.
- Meetings in which both countries are coded as actors have **no recipient** and are not edges.
- Roles and event types are LLM-coded and can be wrong; the case pages show the quotes so they can be checked.
- "Intermediary" in the Sankey is a reported pattern (a country that gives to the party and also receives aid in the
  same period). Big donors such as the USA appear as intermediaries partly because they are also reported receiving
  some aid; page 3 tests whether a path is more than that.
- The matched null model has a tolerance (8% of windows). Stronger conditioning also removes real coordination.
- The matching on activity is an association, not proof that words cause aid.
- 60 countries; the long tail is not in the graph.

---

## 2b. Angle three: support clusters (page 2)

Blocks are found with Leiden community detection on a directed graph per layer (actions or words) and category
(cooperation or conflict), per period (whole period or quarter), on event counts or log(1 + n), with or without the biggest
countries. `src/clusters.py` has the method, `scripts/build_cluster_results.py` precomputes everything (about 9 minutes,
no API calls, 248 clusterings).

- **Stability.** 100 runs, each on re-drawn events (Poisson) and with its own seed. A country is *certain* if, on average,
  it shares a block with its blockmates in at least 70% of the runs. Seeds alone would not show uncertainty: Leiden settles
  a tie the same way every time.
- **Null model.** A directed configuration model (1,000 draws) that keeps the number of events every country gives and
  receives. Modularity, the biggest block and the bridge score of each country are compared with it.
- **Brokers.** The participation coefficient (how evenly a country spreads its ties over blocks), shown only for certain
  countries and only called "standing out" if clearly above the null model.
- **Words versus actions.** Adjusted Rand index and NMI between the two partitions with a permutation null and a bootstrap
  interval; per-country deviation scores from the co-assignment matrices; a 2 × 2 of country pairs; a held-out test of whether
  words blocks from the first two thirds predict material cooperation in the last third (matched on activity).
- **Case studies** (page 5) use the blocks: the party's block in words and in actions, who bridges, how that moved over
  the quarters. The fourth case is chosen by the data: the two countries whose words and actions disagree most.

What the data says (as of 18 October 2026):

- **Four action blocks:** a Western block around Ukraine (USA, Ukraine, UK, Germany, Canada, France, Japan, Australia,
  Poland), a Eurasian block (Russia, Iran, India, China, Pakistan, Afghanistan, North Korea), an Israel–Palestine block (with
  UAE, Lebanon, Egypt, Jordan) and a Syria–Gulf block (Saudi Arabia, Qatar, Türkiye, Iraq, Yemen). Modularity 0.21 against
  0.02 for the null model. Without the USA the blocks barely change (agreement 0.73). Only 37 of 60 countries have a certain
  block; quarter by quarter, material cooperation is too sparse for most countries.
- **Words give three blocks** and agree only partly with the action blocks (adjusted Rand 0.22, 95% interval 0.14 to 0.36 for
  single re-drawn clusterings; chance 0.00).
- **Adversaries do not sit in opposite blocks.** 37% of conflict events are between countries of the same cooperation
  block, against 26% by chance. The blocks are regional: neighbours cooperate and fight.
- **Words blocks predict later material cooperation only weakly:** +6.7 points (95% interval +0.5 to +12.8) for all material
  cooperation, no clear effect for new aid links (+2.1, interval −2.6 to +6.9).
- **AID is wide in GENOME.** It also holds humanitarian exchanges and military help (Russia returning soldiers' bodies to
  Ukraine is coded "Russia AID Ukraine"). The case studies show the events so this can be checked.

---

## 2c. Data preparation: what was fetched, checked, cleaned and chosen

The assignment asks to "retrieve and prepare the GENOME data for your chosen angle". The choices, in order:

**1. Retrieve: counts first, events only where needed.** A free query returns an exact `total` for any filter, and page 1 holds
all events when `total` is at most 250. So each (actor, recipient, event type) cell of the 60 biggest countries is cut into
time windows until every window has at most 250 events, and those windows are read from page 1 (`scripts/build_edge_counts.py`).
Result: 50,487 events with date, type and intensity, 63,474 country-pair edges.

**2. Verify the one claim everything rests on, with credits.** "A window of at most 250 events is complete on the free page"
was tested once against the paid route: one query of 700 events (Russia → Ukraine, ASSAULT, 2024-01-01 to 2026-01-26) was
read the way a paying user would (page 1 without the key, pages 2 and 3 with it), and again as 9 free windows. The two sets are
**identical, id by id and field by field, without duplicates**. Cost: **450 credits** (9,550 left); the result is stored in
`data/processed/slicing_verification.json`. Before spending, the API client's budget guard was changed to refuse *before* a paid
request is sent (it used to check afterwards).

**3. Enrich with text.** Every closed window was read once more to keep what the small stored fields lack: summary, verbatim
quote, names of actor, recipient and third party, locations, article links
(`scripts/build_event_text.py`, 50,487 events, 14.5 MB, about a minute, free).

**4. Clean: AID is a mixed class.** GENOME's AID holds military help, money and humanitarian relief, but also exchanges of
prisoners and bodies (Russia returning soldiers' bodies is coded "Russia AID Ukraine"), evacuations of citizens, trade measures
and ceremonies that only mention aid. `src/aid_types.py` splits the 3,161 AID events by readable text rules:
military 1,012, humanitarian 504, financial 350, **support in total 59%**; other 767, trade 327, exchange 118, evacuation 83.
Accuracy against hand labels: **78% for the category and 85% for "support or not" on a fresh sample of 40 labelled before the
rules were run on it**; 92% on the 40 used to tune the rules (not an honest number). The exchanges, the target of the cleaning,
were found 4 of 4. Both label sets are in `tests/`.

**5. Choose the definition that is most stable** (`scripts/compare_variants.py`). Three definitions of the action layer, each
clustered for the whole period and for 2025, and each perturbed (damped counts, without the biggest country, finer and coarser
resolution):

| Definition | Events | Countries with a firm block | Robust against other settings (agreement with the original blocks) | 2025 blocks agree with the whole period |
|---|---|---|---|---|
| Paper's layer (AID, cooperation, de-escalation) | 5,055 | 31 | 0.53 | 0.31 |
| Same, but AID only if it is support | 3,494 | 26 | 0.52 | 0.58 |
| **Support only** (military, humanitarian, financial AID) | 2,163 | 33 | **0.71** | 0.52 |

Cleaning alone does not help; keeping *only* support does, at the price of 57% of the events. **Support only is the default
action layer on page 2**; the paper's layer stays selectable. Same four blocks either way (West/Ukraine, Eurasian,
Israel–Palestine, Syria–Gulf), with Russia a firm member of the Eurasian block only under support only. Checked on 2025 alone,
three of the four blocks hold (West/Ukraine, Israel–Palestine, Syria–Gulf); the Eurasian block does not hold in a single year,
and the USA becomes uncertain. Windows shorter than a year are not reliable for actions at all (best half-year: 22 countries with a
firm block, the whole period: 37), so the dashboard analyses the whole period with 2025 as the check.

---

## 3. The dashboard

| Page | Content |
|---|---|
| **1 · Starting analysis** | A) which conflict (most hostile pairs); B) movement over time: lines, animation on the axis, animation of words vs actions; C) where everyone stands, over a period; D) words-versus-actions quadrant with filters, names and search |
| **2 · Support clusters** | A) maps of the blocks in words and in actions, side by side, with flows between blocks and the countries where they disagree; brokers and flows; words versus actions (agreement, deviation, 2 × 2, candidate latent ties, prediction test); how sure we are (null model, resolution, hub removal, quarterly stability); blocks over time; chains inside and between blocks |
| **3 · The map** | World map by month; colour by events or conflict share; click a country for its profile and newest events |
| **4 · Support chains** | A) map of reported support (arcs, period, follow one country two steps back and forward); B) are the chains more than volume; C) do words come before support, 2 × 2, candidate latent paths |
| **5 · Case studies** | Four cases: the blocks around a party, who supplies it (Sankey with time slider), the events behind a path and an optional AI summary |

Some sections have a box for the author's own words (`src/journey_text.py`); left empty, it is simply not shown.

---

## 4. Key design choices and trade-offs

- **Free queries first.** `total` on the free page 1, with filters on actor, recipient, event type, category and date,
  gives exact counts at no cost. The trade-off is no per-event intensity and no order within a window. The 10,000 credits
  are reserved for questions that need those (page 6); a single check that the free route is complete used 450 of them
  (section 2c).
- **Dyads, not countries.** An early country-level "acts per word" ratio mixed friendly and hostile relationships; it was
  dropped.
- **Event categories follow the paper**, including debatable ones: a *request* counts as a hostile word and a
  *consultation* (31% of all events) as a friendly one. The page warns about this.
- **Aid only as the default for "support".** 74% of material cooperation is AID; COOPERATE is mutual and RETREAT is
  de-escalation (for example prisoner exchanges), so "Ukraine → Russia" in the broad layer is mostly not support.
- **Rolling windows** for time: weekly counts per country are tiny, so a score uses 2–12 weeks and moves one week at a
  time. Only the biggest players (USA, UK) have enough events in a 2-week window. Gaps mean *not enough events*, never
  *neutral*.
- **Precompute the heavy statistics** (`scripts/build_chain_results.py`) so the dashboard never runs thousands of
  permutations.
- **Processed data is committed** (about 16 MB) so the dashboard runs after a clone without any key; the raw API cache
  (`data/cache/`) is not.

---

## 5. How to run

### Requirements

- macOS or Linux, **Python 3.12** (pinned in `.python-version`; 3.14 was avoided because many libraries lag behind it)
- Internet access to `genome.app.hcss.nl` (only to rebuild the data or to fetch the events behind a path)
- An API key is **never needed** by the dashboard. It is only used by `scripts/check_api.py` and the planned escalation
  ladder. `ANTHROPIC_API_KEY` is only needed for the optional AI summary.

### Setup

```bash
brew install python@3.12                                   # or any other way to get Python 3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt                  # pinned versions
.venv/bin/pip install -r requirements-dev.txt              # only to run the tests

cp .env.example .env                                       # optional: paste your genome_… key (git-ignored)
.venv/bin/python scripts/check_api.py                      # optional: connection test and credit balance, costs nothing
```

### Launch

```bash
.venv/bin/streamlit run app.py          # opens http://localhost:8501
```

If you change a file in `src/` while the app is running, restart it: Streamlit reloads `app.py` but not imported modules.

### Rebuild the data (free, resumable; about 2 hours)

The committed files in `data/processed/` are enough to run the dashboard. To rebuild everything from the public API:

```bash
zsh scripts/build_all.sh
```

It runs nine steps: monthly counts, hostile pairs, quarterly and weekly conflict data, the edge counts, the event text, the chain
statistics, the cluster results and the comparison of definitions. It never spends credits. Every step can be rerun on its own; the edge-count script defaults to a dry run that prints the
estimated requests and minutes (`--run` to build).

### Tests

```bash
.venv/bin/pytest -q                  # 222 tests, about 2 minutes (the slow ones run the whole app on the real data)
.venv/bin/pytest -q -m "not slow"    # unit tests only, a few seconds
.venv/bin/python scripts/mutation_check.py   # breaks the code on purpose in 50 places; every break must be caught
```

The unit tests use small synthetic worlds with a known answer (a planted effect, a pure volume artefact, a shared war
period). The mutation check makes sure the tests would notice if the code were wrong. It turns Python bytecode caching
off, because a cached copy of the broken file can otherwise shadow the restored one.

---

## 6. What lives where

```
app.py                         Thin shell: page config, data loading, sidebar, picks a page
views/                         One module per page
  analysis.py                  Page 1 (reliability)
  map_page.py                  Page 2
  clusters_page.py             Page 2 (support clusters)
  chains_page.py               Page 4 (support chains)
  case_page.py                 Page 5 (case studies)
  common.py                    Shared loaders, constants, the "how to read this" note
src/
  taxonomy.py                  THE grouping of the 16 event types into words/actions (one constant)
  genome_api.py                API client: free counts, cached pages, credit budget guard, free-only mode
  metrics.py                   Reliability metrics: stances, side scores, severity weights, rolling windows
  edge_counts.py               Adaptive time windows (split only while > 250 events) and compact events
  chains.py                    Graph, chain weights, null models, false discovery rate, brokers, cycles
  aid_types.py                 Splits AID events into military, humanitarian, financial, exchange, evacuation, trade, other
  support_events.py            Events with their AID kind; the 'support only' action layer
  clusters.py                  Leiden blocks, stability, brokers, configuration-model null, words-vs-actions agreement
  cluster_tables.py            Block names, brokers, candidate latent ties, membership over time
  hidden.py                    Words → material test (matching, bootstrap), 2×2 classes, latent ties
  geo.py                       Country centroids and directed arcs for the map
  narrative.py                 Optional Claude summary of the events on screen
  journey_text.py              The author's own text per section
scripts/
  build_all.sh                 Rebuild all data, in order
  build_monthly_counts.py      Monthly counts per country and event type
  find_conflict_pairs.py       The most hostile country pairs
  build_dyad_counts.py         Quarterly counts toward both parties of a conflict
  build_weekly_counts.py       Weekly counts for the 30 biggest players per conflict
  probe_chains.py              Step-0 probe: what can the free API do? (writes a report)
  build_edge_counts.py         Every event between the 60 biggest countries (free-only client)
  build_chain_results.py       Chain statistics, brokers, cycles, words → aid test, classes
  build_event_text.py          Text, names, locations and links for every stored event (free)
  verify_slicing.py            Compares paid paging with free slicing on one query (spends ~450 credits; dry run by default)
  compare_variants.py          Which definition of 'support' gives the most stable blocks
  build_cluster_results.py     All cluster results (248 clusterings), agreement, deviation, prediction
  probe_clusters.py            Step-0 probe for the cluster work (writes a report)
  mutation_check.py            Breaks the code on purpose to test the tests
  build_extra_conflicts.sh     Optional: add more conflicts to page 1
  check_api.py                 Connection and credit-balance check
  explore_*.py                 First feasibility tests (kept as part of the journey)
tests/                         Unit tests per module and smoke tests of the whole app
data/processed/                Generated data (committed); data/cache/ holds API response caches (not committed)
assets/                        HCSS logo and the world-borders GeoJSON (Natural Earth, public domain)
documents/                     The assignment, the GENOME paper and the API documentation
```

---

## 7. Challenges encountered

- **No event-level data on the free tier**, which first looked like a hard limit. The probe showed that a window with at
  most 250 events *is* complete on page 1, and that nothing here exceeded 250, so the free tier turned out to carry the
  whole analysis.
- **Fuzzy text filters.** `actor_country=US` matches far more than the United States, and `USA` matches nothing. All
  queries use the exact `*_country_id` filters.
- **Volume and shared busy weeks.** The first null model made specific chains look coordinated whenever the same war
  week made everything busy. Hard bins of activity did not help (the result depended on the number of bins); a matched
  shuffle with a tolerance behaves smoothly.
- **A matching that cannot work.** For "all pairs", pairs with words are far more active than any pair without; balance
  stays at 0.8–0.9. The dashboard now says so and points to the new-link comparison.
- **Dependency risk.** The first environment used Python 3.14 with the newest of everything; it was rebuilt on Python
  3.12 with pinned versions.
- **Python bytecode caching.** Our mutation checks sometimes ran against a cached broken copy; the mutation script now
  disables caching.
- **A first measure that did not mean anything.** The country-level "acts per word" ratio; the dyadic design replaced it.

## 8. Ideas for future work

- **Escalation ladder** with the credits: order events within a window, measure transitions between intensity bands and
  the time from threat to action. Page 1 and 3 can only say that two things happened in the same or the next window.
- **Event-level follow-through:** whether a specific AGREE or THREATEN is followed by the matching action toward the
  same recipient within N days.
- **Organisations and co-actor meetings** as nodes and edges once the data gives them an id.
- **Multilingual sources**, which would reduce the bias toward English-language reporting.
- **Hypergraph structure:** joint events with several actors and recipients are split into pairs here.
- **Generative AI:** a classifier for borderline cases (a *request* that is cooperative versus one that is a demand).
- **Scale and schema.** The full corpus is about 140,000 events and grows weekly. A suitable layout is a fact table
  `event(id, date, type, category, intensity, article_count)`, a bridge table `event_edge(event_id, actor_id,
  recipient_id)` and a dimension table `country(id, name, iso)`, with materialised weekly aggregates. The files built
  here are exactly such tables (`events.csv`, `event_edges.csv`); with a database the build scripts become incremental
  jobs that only process the newest week after each Monday refresh of GENOME.

---

## 9. Data and credits

GENOME is an HCSS Datalab dataset (Dell'Orto and Kommandeur, ACL 2026 workshop). This repository only reads from the
public API; see `documents/` for the paper and the API documentation. **Credits used by everything in this repository:
450 of 10,000** (one verification, `scripts/verify_slicing.py`; everything else is free).
