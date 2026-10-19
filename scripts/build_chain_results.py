"""
build_chain_results.py: run the support-chain analysis once and store the results for the dashboard.

Input : data/processed/edges/  (from scripts/build_edge_counts.py)
Output: data/processed/chains/
    chains.csv        every chain A -> B -> C with weight >= 3, per definition / kind / window / null model
    summary.csv       per configuration: how many chains were tested, how many pass, how many chance would give
    brokers.csv       pass-on score per country (aid, 4-week windows)
    cycles.csv        cycles A -> B -> C -> A found in the aid graph
    prediction.csv    the held-out words -> material test on a grid of settings
    classes.csv       every ordered pair in one of four classes (words / material)
    meta.json         parameters and the date of the run

Definitions
    aid       only AID events (directed support; the purest reading of "A supports B")
    material  AID + COOPERATE + RETREAT (the full material-cooperation layer of the paper)
Kinds
    support      cooperation A -> B followed by cooperation B -> C
    to_conflict  AID A -> B followed by ASSAULT B -> D (support that is followed by an attack)
Null models
    shuffle   each edge's timing shuffled on its own (corrects for volume only)
    matched   each edge only moves to windows of similar overall activity (also corrects for shared busy weeks)

No API calls and no credits: it only reads the stored edge files. Takes about 10-15 minutes.

Run (project root):  .venv/bin/python scripts/build_chain_results.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chains as ch   # noqa: E402
import hidden as hd   # noqa: E402

EDGES = ROOT / "data" / "processed" / "edges"
OUT = ROOT / "data" / "processed" / "chains"
DEFINITIONS = {"aid": ["AID"], "material": None}          # None = every material cooperation type
WINDOWS = (2, 4, 8)
NULLS = ("shuffle", "matched")
MIN_WEIGHT = 3
N_PERM, N_REFINE = 1000, 20000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    leaves = pd.read_csv(EDGES / "leaf_counts.csv")
    events = pd.read_csv(EDGES / "events.csv")
    edges = pd.read_csv(EDGES / "event_edges.csv")
    meta = json.loads((EDGES / "meta.json").read_text())
    names = {int(k): v for k, v in meta["countries"].items()}
    nodes = list(names)
    idx = {c: i for i, c in enumerate(nodes)}
    start, end = date.fromisoformat(meta["since"]), date.fromisoformat(meta["latest"])

    def graph(w, layer, cat, types=None):
        return ch.build_graph(events, edges, leaves, nodes, start, end, w, layer, cat, event_types=types).counts

    def test(X, Y, null):
        res = ch.chain_significance(X, Y, n_perm=N_PERM, seed=1, null=null)
        t = ch.chains_table(res, nodes, names, min_weight=MIN_WEIGHT)
        cand = t[t.p <= 0.05]
        if len(cand):
            tri = np.array([[idx[a], idx[b], idx[c]] for a, b, c in zip(cand.a, cand.b, cand.c)])
            t.loc[cand.index, "p"] = ch.refine_p_values(X, Y, tri, n_perm=N_REFINE, seed=2, null=null)
        t["q"] = ch.benjamini_hochberg(t.p.values)
        return t

    all_chains, summary = [], []
    t0 = time.time()
    for dname, types in DEFINITIONS.items():
        for kind in ("support", "to_conflict"):
            for w in WINDOWS:
                X = graph(w, "material", "cooperation", types)
                Y = X if kind == "support" else graph(w, "material", "conflict", ["ASSAULT"])
                for null in NULLS:
                    t = test(X, Y, null)
                    for k, v in (("definition", dname), ("kind", kind), ("window_weeks", w), ("null", null)):
                        t[k] = v
                    all_chains.append(t)
                    summary.append({"definition": dname, "kind": kind, "window_weeks": w, "null": null,
                                    "events_in_graph": int(X.sum()), "edges": int((X.sum(axis=2) > 0).sum()),
                                    "tested": len(t), "p_le_0.01": int((t.p <= 0.01).sum()),
                                    "expected_by_chance": round(len(t) * 0.01, 1), "q_le_0.10": int((t.q <= 0.10).sum())})
                    print(f"{dname:9} {kind:11} W={w} {null:8} tested {len(t):5,} q<=0.10: {(t.q <= 0.10).sum():4} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.concat(all_chains).to_csv(OUT / "chains.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)

    # brokers and cycles: aid graph, 4-week windows
    X = graph(4, "material", "cooperation", ["AID"])
    brokers = []
    for null in NULLS:
        b = ch.broker_significance(X, n_perm=N_PERM, seed=1, null=null)
        df = pd.DataFrame({"country_id": nodes, "country": [names[c] for c in nodes], "score": b["score"],
                           "expected": b["expected"], "z": b["z"], "p": b["p"], "null": null})
        df["q"] = ch.benjamini_hochberg(df.p.values)
        brokers.append(df)
    pd.concat(brokers).to_csv(OUT / "brokers.csv", index=False)
    cyc = ch.cycle_significance(X, n_perm=500, seed=1, null="matched")
    A, B, C = np.nonzero(cyc["weight"] >= 1)
    cycles = pd.DataFrame({"a": [nodes[i] for i in A], "b": [nodes[i] for i in B], "c": [nodes[i] for i in C],
                           "weight": cyc["weight"][A, B, C], "expected": cyc["expected"][A, B, C],
                           "p": cyc["p"][A, B, C]})
    cycles["cycle"] = cycles.a.map(names) + " → " + cycles.b.map(names) + " → " + cycles.c.map(names) + " → " + cycles.a.map(names)
    cycles.sort_values("weight", ascending=False).to_csv(OUT / "cycles.csv", index=False)

    # words -> material
    w4 = 4
    words = graph(w4, "words", "cooperation")
    everything = sum(graph(w4, layer, cat) for layer in ("words", "material") for cat in ("cooperation", "conflict"))
    split = int(words.shape[2] * 2 / 3)
    rows, classes = [], []
    for mname, mtypes in DEFINITIONS.items():
        material = graph(w4, "material", "cooperation", mtypes)
        for only_new in (False, True):
            for k in (2, 3, 5):
                for out in (1, 3):
                    r = hd.prediction_test(words, material, everything, split, k_verbal=k, outcome_min=out,
                                           only_new=only_new)
                    row = {"material": mname, "only_new": only_new, "k_verbal": k, "outcome_min": out,
                           "n_treated": r["n_treated"], "n_pool": r["n_pool"], "ok": r["ok"]}
                    if r["ok"]:
                        row.update({k_: r[k_] for k_ in ("rate_treated", "rate_matched", "rate_all_untreated",
                                                         "difference", "ci_low", "ci_high")})
                        row["max_abs_smd_after"] = float(r["balance"].smd_after.abs().max())
                    rows.append(row)
        c = hd.classify_pairs(words, material)
        c["material_definition"] = mname
        classes.append(c.assign(a_id=[nodes[i] for i in c.a], r_id=[nodes[i] for i in c.r]))
    pd.DataFrame(rows).to_csv(OUT / "prediction.csv", index=False)
    pd.concat(classes).to_csv(OUT / "classes.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "built": str(pd.Timestamp.now()), "since": str(start), "latest": str(end), "min_weight": MIN_WEIGHT,
        "permutations": N_PERM, "refined_permutations": N_REFINE, "split_window": split,
        "windows_weeks_prediction": w4, "countries": {str(k): v for k, v in names.items()}}, indent=1))
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
