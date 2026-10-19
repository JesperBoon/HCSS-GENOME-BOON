"""
compare_variants.py: which definition of 'support' gives the most stable blocks?

Three definitions of the action-cooperation graph, each clustered for the whole period and for 2025 only:
  V0 baseline            AID + COOPERATE + RETREAT, as in the paper's material-cooperation layer
  V1 cleaned             like V0, but an AID event only counts if its text says military, humanitarian or financial aid
                         (exchanges of prisoners and bodies, evacuations, trade measures and ceremonies are removed)
  V2 support only        only those support AID events: no COOPERATE and no RETREAT

For each: number of blocks, countries with a firm block, mean certainty, strength against the null model, and how much the
blocks change when the setting changes (damped counts, without the biggest country, finer and coarser resolution).
Output: data/processed/clusters/variants.csv. No API calls.

Run (project root):  .venv/bin/python scripts/compare_variants.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chains as ch        # noqa: E402
import clusters as cl      # noqa: E402
import support_events as se   # noqa: E402

E = ROOT / "data" / "processed" / "edges"
OUT = ROOT / "data" / "processed" / "clusters" / "variants.csv"
STABLE = 0.7


def load():
    ev = se.load_events()
    ed = pd.read_csv(E / "event_edges.csv")
    meta = json.loads((E / "meta.json").read_text())
    names = {int(k): v for k, v in meta["countries"].items()}
    return ev, ed, names, meta


def variant_events(ev, name):
    """Events and event types of a variant (V0, V1, V2); support AID events carry the type AID_SUPPORT."""
    if name == "V0 baseline":
        return ev.assign(event_type=ev.event_type.replace({se.SUPPORT_TYPE: "AID"})), ["AID", "COOPERATE", "RETREAT"]
    if name == "V1 cleaned":
        return ev, [se.SUPPORT_TYPE, "COOPERATE", "RETREAT"]
    return ev, [se.SUPPORT_TYPE]


def run(counts, seed):
    """Consensus, null comparison and robustness for one matrix."""
    main = cl.consensus(counts, 1.0, 100, seed)
    null = cl.null_distribution(counts, 500, 1.0, False, seed + 1)
    cmp_ = cl.compare_with_null(main, counts, null)
    hub = cl.hubs(counts, 1)
    perturb = {
        "damped": cl.consensus(counts, 1.0, 40, seed + 2, log=True).membership,
        "without biggest": cl.consensus(cl.without(counts, hub), 1.0, 40, seed + 3).membership,
        "finer": cl.consensus(counts, 1.5, 40, seed + 4).membership,
        "coarser": cl.consensus(counts, 0.75, 40, seed + 5).membership}
    ari = {k: cl.adjusted_rand(main.membership, v) for k, v in perturb.items()}
    return main, cmp_, ari


def main():
    ev, ed, names, meta = load()
    nodes = list(names)
    periods = {"whole period": (meta["since"], meta["latest"]), "2025": ("2025-01-01", "2025-12-31")}
    rows, memberships = [], {}
    for pname, (a, b) in periods.items():
        for vname in ("V0 baseline", "V1 cleaned", "V2 support only"):
            ev2, types = variant_events(ev, vname)
            W = cl.weights_matrix(ch.pair_weights(ev2, ed, types, a, b), nodes)
            m, c, ari = run(W, 7)
            memberships[(pname, vname)] = m.membership
            rows.append({"period": pname, "variant": vname, "events": int(W.sum()), "pairs": int((W > 0).sum()),
                         "blocks": len(set(m.membership[m.membership >= 0])), "firm_countries": int((m.stability >= STABLE).sum()),
                         "mean_certainty": float(np.nanmean(m.stability)), "z_vs_null": c["modularity_z"],
                         **{f"robust_{k}": v for k, v in ari.items()}, "mean_robustness": float(np.mean(list(ari.values())))})
    df = pd.DataFrame(rows)
    for pname in periods:
        ref = memberships[(pname, "V0 baseline")]
        df.loc[df.period == pname, "ari_with_V0"] = [cl.adjusted_rand(memberships[(pname, v)], ref) for v in df[df.period == pname].variant]
    # does 2025 give the same blocks as the whole period, per variant?
    for v in df.variant.unique():
        df.loc[df.variant == v, "ari_2025_vs_whole"] = cl.adjusted_rand(memberships[("2025", v)], memberships[("whole period", v)])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 240)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
