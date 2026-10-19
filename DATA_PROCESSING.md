# How the data was processed

This describes what the code in this repository actually does to turn raw GENOME events into
what the dashboard shows, and why each step exists. It follows the real pipeline, file by file.

## a) The raw data

GENOME turns news articles into structured events: an **actor**, a **recipient**, an **event
type** (one of 16, e.g. `ASSAULT`, `AID`, `AGREE`, `ACCUSE`), a **date**, a **category**
(cooperation or conflict) and an **intensity** score. One row is one reported event — for
example, "Iranian leaders threaten Israel" becomes one row with actor = Iran, recipient =
Israel, type = `THREATEN`, category = conflict.

The free API returns an exact `total` for any filter, and the first page of results holds every
matching event when that total is 250 or fewer. `src/edge_counts.py` uses exactly this: for the
60 biggest countries, every (actor, recipient, event type) combination is queried over the whole
period, and only split into quarters, then months, then weeks, while a window still holds more
than 250 events. A window at or under that limit is read in full from page 1 — for free, with no
API credits spent. The result: 50,487 events with date, type and intensity, and 63,474
country-pair edges, stored as `data/processed/edges/events.csv` and `event_edges.csv`.

## b) Cleaning, filtering and normalisation actually applied

- **Only the 60 biggest countries** are tracked at event level (`edge_counts.py`). Everyone else
  only appears in the coarser monthly counts (`data/processed/monthly_counts.csv`).
- **Organisations (EU, NATO, UN) have no country id** in GENOME's `actor`/`recipient` fields, so
  they are silently dropped wherever a country id is required — they cannot appear as nodes in
  any map or block.
- **Meetings where both countries are coded as actors** have no recipient and are therefore not
  an edge; they do not enter the pair-count data at all.
- **A minimum-events rule everywhere a ratio or score is shown.** A country with 1 cooperative
  event and 0 conflict events is not "100% peaceful," it is a country we barely heard from. Every
  stance, side-score and ratio in this app (`src/metrics.py`) is only computed once both sides of
  it clear a minimum count (default 3, `min_n` in `side_table`); below that, the value is `NaN`
  and the country is left out rather than shown as neutral.
- **AID is a mixed category and gets split.** GENOME's `AID` type covers military aid, money,
  humanitarian relief, but also prisoner/body exchanges, evacuations, trade measures and
  ceremonies that only mention aid. `src/aid_types.py` classifies each `AID` event's text
  (summary + core sentence) into military / humanitarian / financial / exchange / evacuation /
  trade / other by rule-based text matching, and `src/support_events.py` relabels the ones that
  are real support as the pseudo-type `AID_SUPPORT`. The dashboard's cluster page defaults to
  counting only `AID_SUPPORT` as material cooperation ("support only"), not the paper's broader
  AID+COOPERATE+RETREAT layer, because support-only clustering was found to be more robust (see
  `scripts/compare_variants.py` and README section 2c).
- **English-language newswire only.** GENOME is built from one provider's corpus; this is a
  ceiling on coverage, not something the code corrects for.

## c) Feature engineering — one line each

All of this lives in `src/taxonomy.py` (the one grouping every other module imports) and
`src/metrics.py` (the formulas).

| Feature | Formula / rule | Where |
|---|---|---|
| **Verbal vs. material** | 16 event types grouped into two layers (`AGREE, CONSULT, SUPPORT, CONCEDE, REQUEST, ACCUSE, REJECT, THREATEN` = words; the rest = actions) | `taxonomy.py` |
| **Cooperation vs. conflict** | Each event type is pre-labelled one or the other (e.g. `AID` = cooperation, `ASSAULT` = conflict) | `taxonomy.py` |
| **Stance(A→T)** | `(cooperation − conflict) / (cooperation + conflict)` events from A to T, range −1..+1 | `metrics.side_table` |
| **Side score** | `stance(A→second party) − stance(A→first party)`, range −2..+2; near 0 = no clear side | `metrics.side_table` |
| **Hostile share** | `conflict events / (conflict + cooperation events)` between a pair, used to rank "biggest conflicts" | `metrics.py` |
| **Blocks** | Leiden community detection on a directed graph of event counts per layer (words/actions) and category; repeated 100× on resampled events to get a per-country certainty (stable if placed consistently ≥70% of runs); compared against a configuration-model null to confirm the blocks beat chance | `src/clusters.py`, `src/cluster_tables.py` |
| **Chains** | A→B then B→C within the same or next time window; weight = the weaker of the two links, summed over windows; tested against a shuffled-timing null model | `src/chains.py` |

### Worked example (real data, Russia–Ukraine, last three quarters)

Poland toward Russia: 45 events, stance = **−0.91** (heavily conflictual).
Poland toward Ukraine: 22 events, stance = **+0.27** (mildly cooperative).
Side score = 0.27 − (−0.91) = **+1.18** → clearly leans toward Ukraine.
This single number is what gets plotted as one bar in Step 1C, or one dot in Step 1D once its
material and verbal components are also split out separately.

## d) Limitations (stated directly in the app's "how to read this" notes)

- **Reported events, not proven flows.** A chain or an aid path is what the news reported, in the
  order the time windows allow — never a confirmed transfer.
- **`AID` is wide.** Even after splitting into `AID_SUPPORT`, a single path can still mix military
  aid, humanitarian relief and symbolic acts (e.g. returning soldiers' remains).
- **A `REQUEST` counts as a hostile word** in this taxonomy. A country asking another for help
  will show "hostile words" next to "friendly actions" — a known source of apparent mismatches
  that is not a real reliability signal.
- **Coverage is uneven.** Smaller or less-reported countries clear the minimum-event threshold
  less often, in fewer quarters — gaps mean *not enough data*, never *neutral* or *uninvolved*.
- **English-language source bias.** Conflicts and relationships covered less by English-language
  newswire are underrepresented regardless of their real-world scale.
