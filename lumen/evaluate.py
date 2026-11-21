"""Offline evaluation on the held-out future window, split by user segment."""
from __future__ import annotations
import json
import time
import numpy as np

from .metrics import hit_at_k, ndcg_at_k, recall_at_k
from .split import build_cases, cutoffs

KS = (5, 10, 20)
CURVE = 50


def model_zoo(rec):
    return {
        "Popularity": rec.baseline("pop"),
        "Item kNN": rec.baseline("knn"),
        "Matrix factorisation": rec.baseline("mf"),
        "Content-based": rec.baseline("content"),
        "Retrieval only (RRF)": lambda h, p=None: rec.rank(h, p, mode="rrf"),
        "Two-stage (GBDT ranker)": lambda h, p=None: rec.rank(h, p, mode="ranker"),
        "Two-stage + MMR": lambda h, p=None: rec.rank(h, p, mode="mmr"),
    }


def run(rec, inter, cfg, out_dir="reports", max_cases=None, verbose=True):
    lo, hi, horizon = cutoffs(inter, cfg)
    cases = build_cases(inter, hi, None, cfg.ctx_len)
    if max_cases:
        cases = cases[:max_cases]
    born = rec.items.born_day.to_numpy()
    new_item = born >= hi
    sp, N = rec.sp, rec.sp.N
    pop_share = (sp.count + 1) / (sp.count + 1).sum()
    zoo = model_zoo(rec)
    res = {}
    for seg in ("warm", "cold"):
        seg_cases = [c for c in cases if c.kind == seg]
        res[seg] = {"n": len(seg_cases), "models": {}}
    for name, fn in zoo.items():
        t0 = time.time()
        acc = {seg: {"rec": {k: [] for k in KS}, "ndcg": {k: [] for k in KS}, "hit": {k: [] for k in KS},
                     "nov": {k: [] for k in KS}, "cov": {k: set() for k in KS}, "curve": np.zeros(CURVE),
                     "new_hits": {k: 0 for k in KS}, "new_total": 0} for seg in ("warm", "cold")}
        for c in cases:
            a, rel = acc[c.kind], set(c.targets)
            ranked = fn(c.history)
            for k in KS:
                a["rec"][k].append(recall_at_k(ranked, rel, k))
                a["ndcg"][k].append(ndcg_at_k(ranked, rel, k))
                a["hit"][k].append(hit_at_k(ranked, rel, k))
                top = ranked[:k]
                a["cov"][k].update(top)
                a["nov"][k].append(float(np.mean([-np.log2(pop_share[i]) for i in top])) if top else 0.0)
                nrel = {i for i in rel if new_item[i]}
                if k == KS[0]:
                    a["new_total"] += len(nrel)
                a["new_hits"][k] += len(set(top) & nrel)
            for j in range(CURVE):
                a["curve"][j] += recall_at_k(ranked, rel, j + 1)
        for seg in ("warm", "cold"):
            a, n = acc[seg], max(res[seg]["n"], 1)
            res[seg]["models"][name] = {
                "at": {str(k): {"recall": float(np.mean(a["rec"][k])), "ndcg": float(np.mean(a["ndcg"][k])),
                                "hit": float(np.mean(a["hit"][k])), "coverage": len(a["cov"][k]) / N,
                                "novelty": float(np.mean(a["nov"][k])),
                                "new_item_recall": a["new_hits"][k] / a["new_total"] if a["new_total"] else 0.0}
                       for k in KS},
                "curve": [float(x / n) for x in a["curve"]],
            }
        if verbose:
            w, cd = res["warm"]["models"][name]["at"]["10"], res["cold"]["models"][name]["at"]["10"]
            print(f"{name:26s} warm R@10={w['recall']:.3f} N@10={w['ndcg']:.3f} | "
                  f"cold R@10={cd['recall']:.3f} N@10={cd['ndcg']:.3f} | {time.time() - t0:.0f}s")

    # retrieval ceiling and latency for the two-stage system
    cand_hit, cand_n, lat = [], [], []
    for c in cases[:600]:
        t0 = time.perf_counter()
        cand = rec.candidates(c.history)
        cand_n.append(len(cand))
        cand_hit.append(len(set(cand) & set(c.targets)) / len(c.targets))
        t1 = time.perf_counter()
        rec.rank(c.history, mode="mmr")
        lat.append((time.perf_counter() - t1) * 1000)
    out = {
        "meta": {"users": int(inter.user_id.nunique()), "items": int(N), "interactions": int(len(inter)),
                 "horizon_days": horizon, "ranker_cutoff": lo, "train_cutoff": hi,
                 "new_items_in_test": int(new_item.sum()), "cold_context_len": cfg.ctx_len},
        "ks": list(KS), "models": list(zoo), "segments": res,
        "retrieval": {"avg_candidates": float(np.mean(cand_n)), "candidate_recall": float(np.mean(cand_hit)),
                      "catalog": int(N)},
        "ranker": rec.report,
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    with open(f"{out_dir}/metrics.js", "w") as f:       # lets dashboard/index.html work from file://
        f.write("window.METRICS=" + json.dumps(out) + ";")
    return out
