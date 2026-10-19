"""
mutation_check.py: do the tests really notice when the code is broken?

For every entry below one line of source is changed on purpose (a "mutation"), the module's tests are run,
and the original is restored. A mutation the tests do not catch means a test is missing or too weak.

Python caches compiled modules and compares file times in whole seconds, so a restored file could be shadowed
by the cached broken version. This script therefore turns bytecode writing off and clears the caches before
every run.

Run (project root):  .venv/bin/python scripts/mutation_check.py [module ...]    e.g. chains hidden
Exit code 1 if any mutation survives or a pattern is no longer found in the source.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS = {
    "metrics": ("src/metrics.py", ["tests/test_metrics.py"], {
        "stance sign flipped": ("return ((pos - neg) / tot).where(tot >= min_n)", "return ((neg - pos) / tot).where(tot >= min_n)"),
        "side score reversed": ('out[f"side_{kind}"] = stance[second_id] - stance[first_id]', 'out[f"side_{kind}"] = stance[first_id] - stance[second_id]'),
        "rolling window ignored": (".rolling(window).sum()", ".rolling(1).sum()"),
    }),
    "genome_api": ("src/genome_api.py", ["tests/test_genome_api.py"], {
        "budget guard removed": ("if paid and self.spent + PAGE_SIZE > self.budget:", "if False:"),
        "free-only page guard removed": ("if page > 1 and self.free_only:", "if False:"),
        "free-only key guard removed": ("if auth and self.free_only:", "if False:"),
    }),
    "edge_counts": ("src/edge_counts.py", ["tests/test_edge_counts.py"], {
        "250 events no longer closed": ("if total <= limit:", "if total < limit:"),
        "empty windows kept": ("        if total == 0:\n            return\n", ""),
        "weeks end on Saturday": ("6 - start.weekday()", "5 - start.weekday()"),
        "self edges allowed": ("if a != r]", "if True]"),
        "clipping removed": ("start, end = max(a, window.start), min(b, window.end)", "start, end = a, b"),
    }),
    "chains": ("src/chains.py", ["tests/test_chains.py"], {
        "weakest link becomes strongest": ('f = {"min": np.minimum,', 'f = {"min": np.maximum,'),
        "chains back to the start allowed": ("return same * not_back, nxt * not_back", "return same, nxt"),
        "order reversed in time": ("f(x[..., :-1], y[..., 1:])", "f(x[..., 1:], y[..., :-1])"),
        "plain shuffle does nothing": ("        return np.take_along_axis(counts, rng.random(counts.shape).argsort(axis=-1), axis=-1)", "        return counts"),
        "matched null ignores activity order": ("    if order is None:\n        return np.take_along_axis", "    if True:\n        return np.take_along_axis"),
        "matched null moves nothing": ("keys = np.arange(counts.shape[-1]) + tolerance * rng.standard_normal(by_rank.shape)", "keys = np.arange(counts.shape[-1]) + 0.0 * rng.standard_normal(by_rank.shape)"),
        "p-value uses > not >=": ("        ge += (w >= obs)\n    mean = total / n_perm\n    std = np.sqrt(np.maximum(sq / n_perm - mean ** 2, 0))\n    z = np.where(std > 0, (obs - mean) / np.where(std > 0, std, 1), np.where", "        ge += (w > obs)\n    mean = total / n_perm\n    std = np.sqrt(np.maximum(sq / n_perm - mean ** 2, 0))\n    z = np.where(std > 0, (obs - mean) / np.where(std > 0, std, 1), np.where"),
        "broker uses max not min": ("return np.minimum(received, given).sum(axis=1)", "return np.maximum(received, given).sum(axis=1)"),
        "cycles counted in every rotation": ("return w * canonical", "return w"),
        "count-only windows not spread": ("row.count / len(days)", "row.count"),
        "events outside period kept": ("m = m[(m.w >= 0) & (m.w < T) & m.actor_id.isin(idx) & m.recipient_id.isin(idx)]", "m = m[m.actor_id.isin(idx) & m.recipient_id.isin(idx)]"),
        "BH without running minimum": ("q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]", "q_sorted = ranked"),
        "refined p uses > not >=": ("ge += weight(Sxp[ia], Syp[ib]) >= obs", "ge += weight(Sxp[ia], Syp[ib]) > obs"),
        "refined p ignores second graph": ("        Syp = Sxp if same_graph else shuffle_edges(Sy, rng, order, tolerance)", "        Syp = Sy"),
        "upstream looks forward": ("rows = pairs[pairs.recipient.isin(frontier) & ~pairs.actor.isin(seen)]\n            nxt = set(rows.actor)", "rows = pairs[pairs.actor.isin(frontier) & ~pairs.recipient.isin(seen)]\n            nxt = set(rows.recipient)"),
        "visited countries revisited": ("frontier, seen = nxt, seen | nxt", "frontier, seen = nxt, {country}"),
        "reach depth ignored": ("for step in range(1, depth + 1):", "for step in range(1, 2):"),
        "pair_weights end date exclusive": ("ev = ev[dates <= pd.Timestamp(until)]", "ev = ev[dates < pd.Timestamp(until)]"),
        "pair_weights ignores type": ("ev = events[events.event_type.isin(event_types)]", "ev = events"),
        "pair_weights since ignored": ("    if since:\n        ev = ev[dates >= pd.Timestamp(since)]", "    if False:\n        ev = ev[dates >= pd.Timestamp(since)]"),
    }),
    "clusters": ("src/clusters.py", ["tests/test_clusters.py"], {
        "modularity ignores the expected term": ("return float(((W - expected) * same).sum() / total)", "return float((W * same).sum() / total)"),
        "resolution ignored": ("resolution_parameter=resolution, n_iterations=-1, seed=seed)", "resolution_parameter=1.0, n_iterations=-1, seed=seed)"),
        "labels not ordered by size": ("order = sorted(zip(labels, counts), key=lambda lc: (-lc[1], np.flatnonzero(membership == lc[0]).min()))", "order = sorted(zip(labels, counts), key=lambda lc: (lc[1], np.flatnonzero(membership == lc[0]).min()))"),
        "no resampling in the stability runs": ("rng.poisson(counts) if resample else counts", "counts"),
        "participation not squared": ("1 - sum((S[i, membership == c].sum() / k) ** 2 for c in labels)", "1 - sum((S[i, membership == c].sum() / k) for c in labels)"),
        "outside share uses only given ties": ("S = W + W.T\n    out = np.full(W.shape[0], np.nan)\n    for i in np.flatnonzero(membership >= 0):", "S = W\n    out = np.full(W.shape[0], np.nan)\n    for i in np.flatnonzero(membership >= 0):"),
        "null keeps nothing (no shuffle)": ("    R = rng.permutation(R)\n", ""),
        "null leaves self-loops": ("    for _ in range(60):", "    for _ in range(0):"),
        "null p-value uses > not >=": ("p_q = (1 + (nq >= obs_q).sum()) / (1 + len(nq))", "p_q = (1 + (nq > obs_q).sum()) / (1 + len(nq))"),
        "hubs sorted ascending": ("return np.argsort(-(W.sum(axis=0) + W.sum(axis=1)))[:k]", "return np.argsort((W.sum(axis=0) + W.sum(axis=1)))[:k]"),
        "conflict counted between instead of within": ("within = (K * (labels[:, None] == labels[None, :])).sum() / total", "within = (K * (labels[:, None] != labels[None, :])).sum() / total"),
        "adjusted rand without chance correction": ("return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else 1.0", "return float(sum_ij / maximum) if maximum else 1.0"),
        "NMI uses the wrong normalisation": ("return float(mi / ((ha + hb) / 2)) if (ha + hb) > 0 else 1.0", "return float(mi) if (ha + hb) > 0 else 1.0"),
        "clusters matched at any overlap": ("if jac >= 0.2:", "if jac >= 0.0:"),
        "cluster labels reused": ("                out[list(s)] = next_label\n                next_label += 1", "                out[list(s)] = 0"),
        "chains between clusters ignore the end cluster": ("mask = in_x[:, None, None] & in_y[None, None, :]", "mask = in_x[:, None, None] & np.ones_like(in_y)[None, None, :]"),
    }),
    "cluster_tables": ("src/cluster_tables.py", ["tests/test_cluster_tables.py"], {
        "blocks named after their quietest members": ('g = g.sort_values("strength", ascending=False)\n        out[int(c)]', 'g = g.sort_values("strength", ascending=True)\n        out[int(c)]'),
        "brokers include unstable countries": ("(members.stability >= min_stability)", "(members.stability >= -1)"),
        "everything stands out": ("bool(r.participation_z >= 2 and r.participation_p < 0.05)", "True"),
        "candidate pairs also in the same action block": ("(mem_words[i] == mem_words[j]) & (mem_actions[i] != mem_actions[j])", "(mem_words[i] == mem_words[j])"),
        "candidate pairs without words kept": ("    i, j, ev = i[ev > 0], j[ev > 0], ev[ev > 0]", "    pass"),
        "labels not carried over between periods": ("    aligned = cl.align_over_time(mats)\n", "    aligned = mats\n"),
        "a change is counted when nothing changed": ("changes += (ok & (aligned[p0] != aligned[p1])).astype(int)", "changes += (ok & (aligned[p0] == aligned[p1])).astype(int)"),
        "sankey ignores certainty": ("        if stability is not None and min_stability > 0:\n            ok &=", "        if False:\n            ok &="),
        "strip ignores certainty": ('np.where(s[col] >= min_stability, (a[col] + 1).astype(str), "?")', 'np.where(s[col] >= -1, (a[col] + 1).astype(str), "?")'),
    }),
    "aid_types": ("src/aid_types.py", ["tests/test_aid_types.py"], {
        "exchange rule removed": ('    ("exchange", r"', '    ("zzz_exchange", r"'),
        "military checked after humanitarian": ('    ("military", r"\\b(weapons?', '    ("zz_military", r"\\b(weapons?'),
        "support includes exchange": ('SUPPORT = ("military", "humanitarian", "financial")', 'SUPPORT = ("exchange", "military", "humanitarian", "financial")'),
        "text of the core sentence ignored": ('text = f"{summary or \'\'} {core_sentence or \'\'}"', 'text = f"{summary or \'\'}"'),
        "case matters": ("re.IGNORECASE | re.DOTALL", "re.DOTALL"),
    }),
    "hidden": ("src/hidden.py", ["tests/test_hidden.py"], {
        "matching ignores covariates": ("        d = ((zc - zt[i]) ** 2).sum(axis=1)", "        d = rng.random(len(zc))"),
        "outcome in the period before": ('t["y"] = (t.material_after >= outcome_min).astype(int)', 't["y"] = (t.material_before >= outcome_min).astype(int)'),
        "treated threshold > not >=": ("treated = t[t.verbal_before >= k_verbal]", "treated = t[t.verbal_before > k_verbal]"),
        "controls may be treated": ("pool = t[t.verbal_before < k_verbal]", "pool = t[t.verbal_before <= k_verbal]"),
        "only_new removed": ("    if only_new:\n        t = t[t.material_before == 0]\n", ""),
        "controls reused": ("        d[~free] = np.inf\n", ""),
        "bootstrap not clustered": ("    uniq, inv = np.unique(clusters, return_inverse=True)", "    uniq, inv = np.unique(np.arange(len(clusters)), return_inverse=True)"),
        "classes swapped": ("[CLASS_BOTH, CLASS_WORDS, CLASS_MATERIAL]", "[CLASS_BOTH, CLASS_MATERIAL, CLASS_WORDS]"),
        "agreement p-value uses > not >=": ("\"ari_p\": float((1 + (null_ari >= ari).sum()) / (1 + n_perm))", "\"ari_p\": float((1 + (null_ari > ari).sum()) / (1 + n_perm))"),
        "agreement null does not shuffle": ("null_ari = np.array([c.adjusted_rand(a, rng.permutation(b)) for _ in range(n_perm)])", "null_ari = np.array([c.adjusted_rand(a, b) for _ in range(n_perm)])"),
        "bootstrap does not re-draw the words layer": ("w = c.leiden_once(c.transform(rng.poisson(counts_words), log), resolution, seed + i)", "w = c.leiden_once(c.transform(counts_words, log), resolution, seed + i)"),
        "bootstrap does not re-draw the actions layer": ("a = c.leiden_once(c.transform(rng.poisson(counts_actions), log), resolution, seed + i)", "a = c.leiden_once(c.transform(counts_actions, log), resolution, seed + i)"),
        "deviation ignores the other layer": ("w_not_a = (Cw * (1 - Ca)).sum(axis=1) / sw", "w_not_a = (Cw * 1).sum(axis=1) / sw * 0"),
        "deviation directions swapped": ("a_not_w = (Ca * (1 - Cw)).sum(axis=1) / sa", "a_not_w = (Cw * (1 - Ca)).sum(axis=1) / sa"),
        "classes: words and actions swapped": ("[CLASS_UNCERTAIN, CLASS_SAME_BOTH, CLASS_SAME_WORDS, CLASS_SAME_ACTIONS]", "[CLASS_UNCERTAIN, CLASS_SAME_BOTH, CLASS_SAME_ACTIONS, CLASS_SAME_WORDS]"),
        "classes ignore stability": ("                    [CLASS_UNCERTAIN, CLASS_SAME_BOTH", "                    [CLASS_UNCERTAIN if False else CLASS_SAME_NEITHER, CLASS_SAME_BOTH"),
        "cluster prediction looks at the earlier period": ("\"y\": (after >= outcome_min).astype(int)})", "\"y\": (before >= outcome_min).astype(int)})"),
        "cluster prediction: treated = different cluster": ("\"treated\": mem_words_p1[i] == mem_words_p1[j],", "\"treated\": mem_words_p1[i] != mem_words_p1[j],"),
        "cluster prediction: only_new removed": ("    if only_new:\n        t = t[t.material_before == 0]\n    treated, pool = t[t.treated]", "    treated, pool = t[t.treated]"),
        "latent ties keep every class": ("    return out[out.ac_class == CLASS_WORDS] if len(out) else out", "    return out"),
    }),
    "geo": ("src/geo.py", ["tests/test_geo.py"], {
        "centroid not area-weighted": ("    cx = ((x + x1) * cross).sum() / (6 * area)\n    cy = ((y + y1) * cross).sum() / (6 * area)", "    cx = x.mean()\n    cy = y.mean()"),
        "first polygon not largest": ("best = max((_ring_centroid(p[0]) for p in polys), key=lambda c: c[2])", "best = _ring_centroid(polys[0][0])"),
        "arcs bow the same way": ("mid_lat, mid_lon = (lat1 + lat2) / 2 - bend * dlon, (lon1 + lon2) / 2 + bend * dlat", "mid_lat, mid_lon = (lat1 + lat2) / 2 + bend * abs(dlon), (lon1 + lon2) / 2 + bend * abs(dlat)"),
        "lat and lon swapped": ('out[feat["properties"][key]] = (best[1], best[0])', 'out[feat["properties"][key]] = (best[0], best[1])'),
    }),
    "narrative": ("src/narrative.py", ["tests/test_narrative.py"], {
        "event cap ignored": ("    shown = events[:max_events]", "    shown = events"),
        "quotes not flattened": ('.replace("\\n", " ")[:300]', "[:300]"),
        "quotes not truncated": ('.replace("\\n", " ")[:300]', '.replace("\\n", " ")'),
        "amounts rule dropped": ('- The dataset has no amounts: never state sums, quantities, or that one country "financed" or "armed" another.\n', ""),
        "empty events allowed": ('    if not events:\n        raise ValueError("no events to describe")\n', ""),
        "wrong model sent": ("model=model, max_tokens=400", "model='other', max_tokens=400"),
    }),
}


def clear_caches():
    for d in ROOT.rglob("__pycache__"):
        if ".venv" not in d.parts:
            shutil.rmtree(d, ignore_errors=True)


def run_tests(tests):
    clear_caches()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([str(ROOT / ".venv/bin/pytest"), "-q", "-x", "-p", "no:cacheprovider", *tests],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        return r.returncode
    except subprocess.TimeoutExpired:
        return -1                      # an infinite loop is a failure of the tests too: counts as caught


def check(name):
    path, tests, muts = MUTATIONS[name]
    src = ROOT / path
    original = src.read_text()
    bad = 0
    try:
        if run_tests(tests) != 0:
            print(f"{name}: the tests fail BEFORE any mutation; fix them first")
            return 1
        for label, (old, new) in muts.items():
            if old not in original:
                print(f"  {name}: {label}: PATTERN NOT FOUND (source changed?)")
                bad += 1
                continue
            src.write_text(original.replace(old, new, 1))
            code = run_tests(tests)
            print(f"  {name}: {label}: {'caught' if code != 0 else 'NOT CAUGHT  <---'}", flush=True)
            bad += code == 0
    finally:
        src.write_text(original)
        clear_caches()
    return bad


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MUTATIONS)
    total = sum(check(n) for n in wanted)
    n_mut = sum(len(MUTATIONS[n][2]) for n in wanted)
    print(f"\n{n_mut} mutations in {len(wanted)} modules; {total} not caught.")
    sys.exit(1 if total else 0)
