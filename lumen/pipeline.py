"""Two-stage recommender: multi-retriever candidate generation, a gradient-boosted ranker,
and an MMR diversity re-rank. Cold start is handled in the retrievers (content, popularity and
fresh-item retrievers need no history) and in the ranker (history length is a feature, and it
is trained on truncated histories)."""
from __future__ import annotations
import pickle
import time
from collections import Counter
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .config import Config
from .retrieval import ItemSpace
from .split import build_cases, cutoffs

FEATS = ["mf", "knn", "content", "pop", "age", "is_new", "n_src",
         "src_mf", "src_knn", "src_content", "src_pop", "src_fresh",
         "rank_mf", "rank_knn", "rank_content", "rank_pop", "cat_aff", "hist_len"]


def _top(vec, valid, n, positive_only=True):
    v = np.where(valid, vec, -np.inf)
    if positive_only:
        v = np.where(v > 1e-9, v, -np.inf)
    n = min(n, len(v) - 1)
    idx = np.argpartition(-v, n)[:n]
    return idx[np.isfinite(v[idx])]


def _ranks(vec, valid):
    order = np.argsort(-np.where(valid, vec, -np.inf), kind="stable")
    r = np.empty(len(vec), dtype=np.float32)
    r[order] = np.arange(len(vec))
    return r


class TwoStageRecommender:
    def __init__(self, items, cfg: Config | None = None):
        self.items, self.cfg = items, cfg or Config()
        self.sp: ItemSpace | None = None
        self.clf = None
        self.report = {}

    # ---- stage 1 + feature building -------------------------------------------------------
    def _build(self, sp, hist, prefs=None):
        s = sp.scores(hist, prefs)
        valid = sp.avail.copy()
        valid[np.asarray(hist, dtype=int)] = False
        sel = {
            "mf": _top(s["mf"], valid, self.cfg.n_cand["mf"]),
            "knn": _top(s["knn"], valid, self.cfg.n_cand["knn"]),
            "content": _top(s["content"], valid, self.cfg.n_cand["content"]),
            "pop": _top(s["pop"], valid, self.cfg.n_cand["pop"]),
            "fresh": _top(np.where(sp.is_new, s["content"] + 1e-3 * sp.rand, -np.inf), valid & sp.is_new,
                          self.cfg.n_cand["fresh"], positive_only=False),
        }
        cands = np.unique(np.concatenate(list(sel.values())))
        flags = {k: np.isin(cands, v).astype(np.float32) for k, v in sel.items()}
        counts = np.zeros(sp.n_cat, dtype=np.float32)
        for i in hist:
            counts[sp.cat[i]] += 1
        for c in prefs or []:
            if c in sp.cat_index:
                counts[sp.cat_index[c]] += 2
        aff = counts / counts.sum() if counts.sum() else counts
        F = np.column_stack([
            s["mf"][cands], s["knn"][cands], s["content"][cands], s["pop"][cands],
            sp.age[cands], sp.is_new[cands].astype(np.float32),
            sum(flags.values()),
            flags["mf"], flags["knn"], flags["content"], flags["pop"], flags["fresh"],
            _ranks(s["mf"], valid)[cands], _ranks(s["knn"], valid)[cands],
            _ranks(s["content"], valid)[cands], _ranks(s["pop"], valid)[cands],
            aff[sp.cat[cands]], np.full(len(cands), len(hist), dtype=np.float32),
        ]).astype(np.float32)
        return cands, F

    def candidates(self, hist, prefs=None):
        return self._build(self.sp, hist, prefs)[0]

    # ---- training -------------------------------------------------------------------------
    def fit(self, inter, max_users=2000, verbose=True):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed if hasattr(cfg, "seed") else 0)
        lo, hi, horizon = cutoffs(inter, cfg)
        t0 = time.time()
        # (1) retrievers fitted at the earlier cutoff generate candidates for the ranker's window
        early = ItemSpace(self.items, fresh_days=cfg.fresh_days).fit(inter[inter.ts < lo], now=lo, until=hi)
        cases = build_cases(inter, lo, hi, cfg.ctx_len)
        cat_names = self.items.category.to_numpy()
        rng.shuffle(cases)
        X, y, groups = [], [], []
        for n, c in enumerate(cases[:max_users]):
            hist, prefs = c.history, None
            if c.kind == "warm" and rng.random() < 0.3:      # teach the ranker what short histories look like
                hist = c.history[: rng.integers(0, 4)]
                if rng.random() < 0.6:                       # ...and what onboarding category picks look like
                    cats = Counter(cat_names[c.history]).most_common(2)
                    prefs = [name for name, _ in cats]
            cands, F = self._build(early, hist, prefs)
            X.append(F)
            y.append(np.isin(cands, c.targets).astype(int))
            groups.append(np.full(len(cands), n))
        X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(groups)
        val = g % 7 == 0                                     # hold out ~14% of users for validation
        self.clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=False, random_state=0)
        self.clf.fit(X[~val], y[~val])
        ap = float(average_precision_score(y[val], self.clf.predict_proba(X[val])[:, 1]))
        sub = np.random.default_rng(0).choice(np.where(val)[0], size=min(25000, int(val.sum())), replace=False)
        imp = permutation_importance(self.clf, X[sub], y[sub], scoring="average_precision",
                                     n_repeats=2, random_state=0)
        self.report = {
            "rows": int(len(y)), "positives": int(y.sum()), "val_average_precision": ap,
            "importance": sorted(zip(FEATS, map(float, imp.importances_mean)), key=lambda t: -t[1]),
        }
        # (2) final retrievers see everything before the test window
        self.sp = ItemSpace(self.items, fresh_days=cfg.fresh_days).fit(inter[inter.ts < hi], now=hi,
                                                                       until=horizon + 1)
        if verbose:
            print(f"trained on {len(y):,} rows ({int(y.sum()):,} positives), "
                  f"val AP={ap:.3f}, {time.time() - t0:.1f}s")
        return self

    # ---- inference ------------------------------------------------------------------------
    def _mmr(self, ids, scores, k, lam):
        s = (scores - scores.min()) / (np.ptp(scores) + 1e-9)
        E = self.sp.X[ids]
        sim = E @ E.T
        maxsim = np.zeros(len(ids))
        chosen, taken = [], np.zeros(len(ids), bool)
        for _ in range(min(k, len(ids))):
            adj = np.where(taken, -np.inf, s - lam * maxsim)
            j = int(np.argmax(adj))
            chosen.append(j); taken[j] = True
            maxsim = np.maximum(maxsim, sim[j])
        return [int(ids[j]) for j in chosen]

    def rank(self, hist, prefs=None, mode="mmr", n=None):
        """Ranked item ids. mode: 'rrf' (retrieval only), 'ranker' (stage 2), 'mmr' (stage 2 + diversity)."""
        n = n or self.cfg.out_len
        cands, F = self._build(self.sp, hist, prefs)
        if len(cands) == 0:
            return []
        if mode == "rrf":
            col = [FEATS.index(f) for f in ("rank_mf", "rank_knn", "rank_content", "rank_pop")]
            score = (1.0 / (60.0 + F[:, col])).sum(1)
        else:
            score = self.clf.predict_proba(F)[:, 1]
        order = np.argsort(-score)[: max(n * 2, n)]
        ids, sc = cands[order], score[order]
        if mode == "mmr":
            return self._mmr(ids, sc, n, self.cfg.mmr_lambda)
        return [int(i) for i in ids[:n]]

    def baseline(self, name):
        """Single-signal baselines for comparison. Ties break by popularity."""
        sp = self.sp
        def fn(hist, prefs=None, n=None):
            n = n or self.cfg.out_len
            s = sp.scores(hist, prefs)
            vec = s["pop"] if name == "pop" else s[name] + 1e-3 * s["pop"]
            valid = sp.avail.copy()
            valid[np.asarray(hist, dtype=int)] = False
            return [int(i) for i in np.argsort(-np.where(valid, vec, -np.inf), kind="stable")[:n]]
        return fn

    def recommend(self, hist, prefs=None, k=10):
        """Top-k with metadata and a human-readable reason for each item."""
        ids = self.rank(hist, prefs, mode="mmr", n=k)
        sp, s = self.sp, self.sp.scores(hist, prefs)
        out = []
        for i in ids:
            reason = "Popular right now"
            if sp.is_new[i]:
                reason = "New release"
            elif len(hist) and s["knn"][i] > 0 and s["knn"][i] >= 0.6 * s["knn"].max():
                via = max(hist, key=lambda h: sp.S[h, i])
                reason = f"Fans of {self.items.title[via]} also liked this"
            elif s["content"][i] > 0.35:
                reason = f"Matches your interest in {self.items.category[i]}"
            r = self.items.iloc[i]
            out.append({"item_id": int(i), "title": r.title, "category": r.category, "reason": reason})
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
