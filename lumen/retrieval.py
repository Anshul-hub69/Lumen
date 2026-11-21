"""Stage 1 building blocks. ItemSpace holds every signal a retriever needs and can score the
whole catalog for any user from a list of history item ids, so brand-new users work by design."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer


def cat_token(name: str) -> str:
    return "cat_" + name.lower().replace(" ", "_")


class ItemSpace:
    def __init__(self, items: pd.DataFrame, k_mf=32, knn_k=50, halflife=30.0, fresh_days=14, seed=0):
        self.items, self.k_mf, self.knn_k = items, k_mf, knn_k
        self.halflife, self.fresh_days, self.seed = halflife, fresh_days, seed

    def fit(self, inter: pd.DataFrame, now: int, until: int):
        """inter must only contain interactions with ts < now.
        Items released before `until` are available; their age is measured at `now`."""
        N = len(self.items)
        self.N, self.now = N, now
        born = self.items.born_day.to_numpy()
        self.avail = born < until
        self.age = np.clip(now - born, 0, 365).astype(np.float32)
        self.is_new = (self.age <= self.fresh_days) & self.avail
        self.rand = np.random.default_rng(self.seed).random(N)
        cats = sorted(self.items.category.unique())
        self.cat_index = {c: i for i, c in enumerate(cats)}
        self.cat = self.items.category.map(self.cat_index).to_numpy()
        self.n_cat = len(cats)

        uid = pd.factorize(inter.user_id)[0]
        R = sp.csr_matrix((np.ones(len(inter)), (uid, inter.item_id.to_numpy())), shape=(uid.max() + 1, N))
        R.data[:] = 1.0
        deg_i = np.asarray(R.sum(0)).ravel()
        deg_u = np.asarray(R.sum(1)).ravel()
        self.count = deg_i

        # popularity with exponential time decay (recent interest counts more)
        w = np.exp(-(now - inter.ts.to_numpy()) / self.halflife)
        decayed = np.bincount(inter.item_id.to_numpy(), weights=w, minlength=N)
        pop = np.log1p(decayed)
        self.pop = (pop / pop.max() if pop.max() > 0 else pop).astype(np.float32)

        # item-item cosine similarity, pruned to the top knn_k neighbours per item
        co = (R.T @ R).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        d = sp.diags(1.0 / np.sqrt(np.maximum(deg_i, 1)))
        S = (d @ co @ d).tocsr()
        rows, cols, vals = [], [], []
        for i in range(N):
            a, b = S.indptr[i], S.indptr[i + 1]
            v, c = S.data[a:b], S.indices[a:b]
            if len(v) > self.knn_k:
                top = np.argpartition(-v, self.knn_k)[: self.knn_k]
                v, c = v[top], c[top]
            rows += [i] * len(v); cols += list(c); vals += list(v)
        self.S = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        # matrix-factorisation item embeddings from a truncated SVD of the normalised matrix
        Rn = sp.diags(1.0 / np.sqrt(np.maximum(deg_u, 1))) @ R @ sp.diags(1.0 / np.sqrt(np.maximum(deg_i, 1)))
        k = max(2, min(self.k_mf, min(Rn.shape) - 1))
        _, s, Vt = svds(Rn.astype(np.float64), k=k, random_state=self.seed)
        Q = Vt.T * np.sqrt(s)
        self.Qn = (Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)).astype(np.float32)

        # content vectors (category + tags, TF-IDF). New items have these before any interaction.
        docs = self.items.category.map(cat_token) + " " + self.items.tags.fillna("")
        tf = TfidfVectorizer(token_pattern=r"[^ ]+")
        self.X = tf.fit_transform(docs).toarray().astype(np.float32)
        self.vocab = tf.vocabulary_
        return self

    def scores(self, hist, prefs=None):
        """Score every catalog item for a user described only by history ids and optional
        onboarding category picks. Returns arrays of length N."""
        h = np.asarray(hist, dtype=int)
        z = np.zeros(self.N, dtype=np.float32)
        mf = knn = z
        p = np.zeros(self.X.shape[1], dtype=np.float32)
        if len(h):
            mf = self.Qn @ self.Qn[h].sum(0) / len(h)
            knn = np.asarray(self.S[h].sum(0)).ravel().astype(np.float32) / len(h)
            p = self.X[h].mean(0)
        for c in prefs or []:
            j = self.vocab.get(cat_token(c))
            if j is not None:
                p = p.copy()
                p[j] += 0.5
        norm = np.linalg.norm(p)
        content = self.X @ (p / norm) if norm > 0 else z
        return {"mf": mf, "knn": knn, "content": content, "pop": self.pop}
