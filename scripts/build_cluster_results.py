"""
build_cluster_results.py: run the support-cluster analysis once and store the results for the dashboard.

Input : data/processed/edges/   (from scripts/build_edge_counts.py; no new API requests are made here)
Output: data/processed/clusters/
    configs.csv               one row per clustering: graph, period, settings, modularity against the null model, stability
    members.csv               one row per country and clustering: cluster, stability, broker scores
    flows.csv                 events between clusters (of the graph itself, and conflict events between cooperation clusters)
    conflict_vs_clusters.csv  do adversaries sit in different blocks?
    sensitivity.csv           how the result changes with the resolution
    agreement.csv             words clusters against action clusters (adjusted Rand, NMI, null model, bootstrap interval)
    deviation.csv             per country: how often its words partners are not its action partners, and the reverse
    prediction.csv            held-out test: do words clusters predict later material cooperation?
    coassign.npz              co-assignment matrices (how often two countries share a cluster in the stability runs)
    meta.json

Graphs   : layer (actions = material events, 'support' = only military, humanitarian and financial AID, words = verbal events)
           x category (cooperation, conflict)
Periods  : the whole period, and every calendar quarter (the last quarter is incomplete)
Settings : raw counts or log(1 + n); with or without the biggest hub country (and without the three biggest, whole period)
Per clustering: 100 Leiden runs on re-drawn events (stability), a consensus partition, and 1,000 draws of a directed
configuration model for the null comparison.

Run (project root):  .venv/bin/python scripts/build_cluster_results.py        (about 10 minutes on 8 cores)
"""

import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chains as ch       # noqa: E402
import clusters as cl     # noqa: E402
import hidden as hd       # noqa: E402
import support_events as se   # noqa: E402
import taxonomy as tx     # noqa: E402

EDGES = ROOT / "data" / "processed" / "edges"
OUT = ROOT / "data" / "processed" / "clusters"
N_RUNS, N_NULL = 100, 1000
RESOLUTIONS = (0.5, 0.75, 1.0, 1.5, 2.0)
STABLE = 0.7


def cid(layer, cat, period, log, hubs, res):
    return f"{layer}|{cat}|{period}|log{int(log)}|hub{hubs}|res{res}"


def run_config(task: dict) -> dict:
    """One clustering with stability, null model and descriptive tables. Runs in a worker process."""
    counts, log, res = task["counts"], task["log"], task["res"]
    drop = task["drop"]
    seed = zlib.crc32(task["cid"].encode()) % (1 << 30)
    C = cl.without(counts, drop) if len(drop) else counts
    cons = cl.consensus(C, res, task["n_runs"], seed, log=log)
    Wt = cl.transform(C, log)
    null = cl.null_distribution(C, task["n_null"], res, log, seed + 1)
    cmp_ = cl.compare_with_null(cons, Wt, null)
    m, n = cons.membership, len(counts)
    act = m >= 0
    sizes = np.bincount(m[act]) if act.any() else np.array([0])
    strength = C.sum(axis=0) + C.sum(axis=1)
    cfg = {"cid": task["cid"], "layer": task["layer"], "category": task["cat"], "period": task["period"],
           "partial": task["partial"], "log": log, "hubs": task["hubs"], "resolution": res,
           "n_active": int(act.sum()), "n_clusters": int(len(sizes) if act.any() else 0),
           "events": int(C.sum()), "pairs": int((C > 0).sum()), "modularity": cons.modularity,
           "observed_quality": cons.observed_quality, "null_mean": cmp_["modularity_null_mean"],
           "z": cmp_["modularity_z"], "p": cmp_["modularity_p"], "largest_share": cmp_["largest_share"],
           "largest_share_null": cmp_["largest_share_null_mean"],
           "mean_stability": float(np.nanmean(cons.stability)) if np.isfinite(cons.stability).any() else np.nan,
           "n_stable": int((cons.stability >= STABLE).sum()), "n_runs": task["n_runs"], "n_null": task["n_null"]}
    members = pd.DataFrame({"cid": task["cid"], "country_id": task["nodes"], "cluster": m, "stability": cons.stability,
                            "participation": cmp_["participation"], "participation_z": cmp_["participation_z"],
                            "participation_p": cmp_["participation_p"], "outside_share": cl.outside_share(C, m),
                            "strength": strength, "dropped": [i in set(drop) for i in range(n)]})
    f = cl.flows(C, m).stack().rename("events").reset_index()
    f.columns = ["from_cluster", "to_cluster", "events"]
    flows = [f.assign(cid=task["cid"], kind="own")]
    cvc = None
    if task["conflict"] is not None:
        K = cl.without(task["conflict"], drop) if len(drop) else task["conflict"]
        r = cl.conflict_between_clusters(m, K, 1000, seed + 2)
        cvc = {"cid": task["cid"], **r}
        fk = cl.flows(K, m).stack().rename("events").reset_index()
        fk.columns = ["from_cluster", "to_cluster", "events"]
        flows.append(fk.assign(cid=task["cid"], kind="conflict_over_cooperation_clusters"))
    return {"config": cfg, "members": members, "flows": pd.concat(flows), "cvc": cvc, "coassign": cons.coassign.astype(np.float32),
            "membership": m, "stability": cons.stability}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = se.load_events()       # stored events with the kind of every AID event; support AID is typed AID_SUPPORT
    edges = pd.read_csv(EDGES / "event_edges.csv")
    meta = json.loads((EDGES / "meta.json").read_text())
    names = {int(k): v for k, v in meta["countries"].items()}
    nodes = list(names)
    start, end = pd.Timestamp(meta["since"]), pd.Timestamp(meta["latest"])
    quarters = pd.period_range(start, end, freq="Q")
    periods = {"all": (start, end, False)}
    for q in quarters:
        periods[str(q)] = (max(q.start_time.normalize(), start), min(q.end_time.normalize(), end), q.end_time.normalize() > end)

    def graph(layer, cat, a, b):
        return cl.weights_matrix(ch.pair_weights(events, edges, se.types_for(layer, cat), str(a.date()), str(b.date())), nodes)

    # 'support' = only military, humanitarian and financial AID (see src/aid_types.py); 'material' = the paper's layer
    G = {(layer, cat, p): graph(layer, cat, a, b) for layer, cat in (("material", "cooperation"), ("material", "conflict"),
                                                                   ("words", "cooperation"), ("words", "conflict"),
                                                                   ("support", "cooperation"))
         for p, (a, b, _) in periods.items()}
    strength_all = sum(G[(l, c, "all")].sum(axis=0) + G[(l, c, "all")].sum(axis=1)
                       for l in ("material", "words") for c in ("cooperation", "conflict"))      # the paper's layers only
    hub_order = [int(i) for i in np.argsort(-strength_all)]
    print("Biggest hubs:", [names[nodes[i]] for i in hub_order[:5]], flush=True)

    tasks = []
    for (layer, cat, p), counts in G.items():
        for log in (False, True):
            for hubs in (0, 1) + ((3,) if p == "all" else ()):
                tasks.append({"cid": cid(layer, cat, p, log, hubs, 1.0), "layer": layer, "cat": cat, "period": p,
                              "partial": periods[p][2], "log": log, "hubs": hubs, "res": 1.0, "counts": counts,
                              "drop": hub_order[:hubs], "nodes": nodes, "n_runs": N_RUNS, "n_null": N_NULL,
                              "conflict": G[("material" if layer == "support" else layer, "conflict", p)] if cat == "cooperation" else None})
    for (layer, cat, p), counts in G.items():                      # resolution sensitivity: whole period, lighter null
        if p != "all":
            continue
        for log in (False, True):
            for res in RESOLUTIONS:
                if res != 1.0:
                    tasks.append({"cid": cid(layer, cat, p, log, 0, res), "layer": layer, "cat": cat, "period": p,
                                  "partial": False, "log": log, "hubs": 0, "res": res, "counts": counts, "drop": [],
                                  "nodes": nodes, "n_runs": 50, "n_null": 200, "conflict": None})
    print(f"{len(tasks)} clusterings on {os.cpu_count()} cores", flush=True)

    results, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, r in enumerate(ex.map(run_config, tasks, chunksize=1), 1):
            results[r["config"]["cid"]] = r
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    configs = pd.DataFrame([r["config"] for r in results.values()])
    configs.to_csv(OUT / "configs.csv", index=False)
    pd.concat([r["members"] for r in results.values()]).to_csv(OUT / "members.csv", index=False)
    pd.concat([r["flows"] for r in results.values()]).to_csv(OUT / "flows.csv", index=False)
    pd.DataFrame([r["cvc"] for r in results.values() if r["cvc"]]).to_csv(OUT / "conflict_vs_clusters.csv", index=False)
    np.savez_compressed(OUT / "coassign.npz", **{k: r["coassign"] for k, r in results.items()})

    # sensitivity to the resolution: how many clusters, how stable, and how alike the partition at resolution 1
    rows = []
    for (layer, cat, log), g in configs[(configs.period == "all") & (configs.hubs == 0)].groupby(["layer", "category", "log"]):
        ref = results[cid(layer, cat, "all", log, 0, 1.0)]["membership"]
        for r in g.itertuples():
            m = results[r.cid]["membership"]
            rows.append({"layer": layer, "category": cat, "log": log, "resolution": r.resolution, "n_clusters": r.n_clusters,
                         "modularity": r.modularity, "mean_stability": r.mean_stability, "n_stable": r.n_stable,
                         "ari_vs_resolution_1": cl.adjusted_rand(m, ref)})
    pd.DataFrame(rows).sort_values(["layer", "category", "log", "resolution"]).to_csv(OUT / "sensitivity.csv", index=False)

    # words clusters against action clusters, per period and setting
    agree, dev = [], []
    for actions in ("material", "support"):
        for p in periods:
            for log in (False, True):
                for hubs in (0, 1) + ((3,) if p == "all" else ()):
                    w = results[cid("words", "cooperation", p, log, hubs, 1.0)]
                    a = results[cid(actions, "cooperation", p, log, hubs, 1.0)]
                    row = {"actions": actions, "period": p, "log": log, "hubs": hubs, **hd.agreement(w["membership"], a["membership"], 1000, 1)}
                    stable = (np.nan_to_num(w["stability"]) >= STABLE) & (np.nan_to_num(a["stability"]) >= STABLE)
                    row["n_stable_both"] = int(stable.sum())
                    row["ari_stable_only"] = cl.adjusted_rand(np.where(stable, w["membership"], -1), np.where(stable, a["membership"], -1))
                    if hubs == 0 and p == "all":
                        boot = hd.agreement_bootstrap(G[("words", "cooperation", p)], G[(actions, "cooperation", p)], 200, 5, 1.0, log)
                        row["ari_boot_low"], row["ari_boot_high"] = np.nanpercentile(boot, [2.5, 97.5])
                    agree.append(row)
                    d = hd.deviation_scores(w["coassign"].astype(float), a["coassign"].astype(float))
                    dev.append(d.assign(actions=actions, period=p, log=log, hubs=hubs, country_id=nodes))
    pd.DataFrame(agree).to_csv(OUT / "agreement.csv", index=False)
    pd.concat(dev).to_csv(OUT / "deviation.csv", index=False)

    # held-out prediction: words clusters in the first two thirds, material cooperation in the last third
    split = start + (end - start) * 2 / 3
    day = pd.Timedelta(days=1)
    p1 = (start, split.normalize())
    p2 = (split.normalize() + day, end)
    all_types = lambda layer, cat: se.types_for(layer, cat)   # noqa: E731
    mat = lambda types, ab: cl.weights_matrix(ch.pair_weights(events, edges, types, str(ab[0].date()), str(ab[1].date())), nodes)   # noqa: E731
    words_p1 = mat(all_types("words", "cooperation"), p1)
    everything_p1 = sum(mat(all_types(l, c), p1) for l in ("words", "material") for c in ("cooperation", "conflict"))
    pred = []
    for log in (False, True):
        for hubs in (0, 1):
            drop = hub_order[:hubs]
            wp1 = cl.without(words_p1, drop) if hubs else words_p1
            cons = cl.consensus(wp1, 1.0, N_RUNS, 11, log=log)
            for mname, mtypes in (("aid", ["AID", se.SUPPORT_TYPE]), ("support", [se.SUPPORT_TYPE]),
                                  ("material", tx.MATERIAL_COOP + [se.SUPPORT_TYPE])):
                m1, m2 = mat(mtypes, p1), mat(mtypes, p2)
                for only_new in (False, True):
                    for out in (1, 3):
                        r = hd.cluster_prediction_test(cons.membership, everything_p1, m1, m2, wp1, only_new, out)
                        pred.append({"log": log, "hubs": hubs, "material": mname, "only_new": only_new, "outcome_min": out, **r})
    pd.DataFrame(pred).to_csv(OUT / "prediction.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "built": str(pd.Timestamp.now()), "since": meta["since"], "latest": meta["latest"], "n_runs": N_RUNS, "n_null": N_NULL,
        "stable_threshold": STABLE, "hubs": [names[nodes[i]] for i in hub_order[:5]], "split_date": str(split.date()),
        "partial_periods": [p for p, v in periods.items() if v[2]], "countries": {str(k): v for k, v in names.items()}}, indent=1))
    print(f"Done in {(time.time() - t0) / 60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
