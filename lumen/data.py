"""Synthetic catalog and interaction log, plus a MovieLens loader."""
from __future__ import annotations
import numpy as np
import pandas as pd

CATEGORIES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller",
              "Animation", "Horror", "Documentary", "Fantasy", "Mystery", "Music"]


def generate(n_users=3000, n_items=1500, horizon=120, seed=7):
    """Latent-factor simulator.

    * Items belong to a category, have a long-tailed quality score and a release day.
      About 6% are released late in the horizon, so they have no history when training
      ends (the item cold-start case).
    * About 10% of users join late (the user cold-start case).
    * Each user picks items with a Gumbel top-k over taste affinity + quality.
    """
    rng = np.random.default_rng(seed)
    C, K = len(CATEGORIES), 8
    cat_vec = rng.normal(size=(C, K))
    item_cat = rng.integers(0, C, n_items)
    item_f = cat_vec[item_cat] + 0.6 * rng.normal(size=(n_items, K))
    quality = np.log(rng.pareto(1.6, n_items) + 1.0)
    new = rng.random(n_items) < 0.06
    born = np.where(new, rng.integers(int(horizon * 0.83), horizon - 5, n_items),
                    rng.integers(0, int(horizon * 0.75), n_items))
    tags = []
    for i in range(n_items):
        home = [f"tag{item_cat[i] * 4 + j:02d}" for j in range(4)]
        t = list(rng.choice(home, size=3, replace=False)) + [f"tag{rng.integers(0, C * 4):02d}"]
        tags.append(" ".join(sorted(set(t))))
    items = pd.DataFrame({
        "item_id": np.arange(n_items),
        "title": [f"{CATEGORIES[c]} Title {i:04d}" for i, c in enumerate(item_cat)],
        "category": [CATEGORIES[c] for c in item_cat],
        "tags": tags,
        "born_day": born,
    })
    user_f = 0.8 * cat_vec[rng.integers(0, C, n_users)] + 0.6 * rng.normal(size=(n_users, K))
    late = rng.random(n_users) < 0.10
    join = np.where(late, rng.integers(int(horizon * 0.83), horizon - 5, n_users),
                    rng.integers(0, int(horizon * 0.5), n_users))
    rows = []
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(2.6, 0.6), 4, 90))
        if late[u]:
            n = int(np.clip(n, 6, 25))
        score = item_f @ user_f[u] / np.sqrt(K) * 1.4 + 0.8 * quality + rng.gumbel(size=n_items)
        chosen = np.argsort(-score)[:n]
        ts = np.sort(rng.integers(join[u], horizon, n))
        for item, t in zip(chosen, ts):
            rows.append((u, int(item), int(max(t, born[item]))))
    inter = pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])
    inter = inter.sort_values(["ts", "user_id"]).reset_index(drop=True)
    return items, inter


def save(items, inter, folder="data"):
    items.to_csv(f"{folder}/items.csv", index=False)
    inter.to_csv(f"{folder}/interactions.csv", index=False)


def load(folder="data"):
    return pd.read_csv(f"{folder}/items.csv"), pd.read_csv(f"{folder}/interactions.csv")


def load_movielens(path, min_rating=4.0):
    """Read a MovieLens folder (ratings.csv, movies.csv) into the same two tables.

    Ratings >= min_rating become positives. Timestamps become days since the first rating,
    and an item's release day is the day of its first positive interaction.
    """
    r = pd.read_csv(f"{path}/ratings.csv")
    m = pd.read_csv(f"{path}/movies.csv")
    r = r[r.rating >= min_rating].copy()
    r["ts"] = ((r.timestamp - r.timestamp.min()) // 86400).astype(int)
    r = r.drop_duplicates(["userId", "movieId"])
    m = m[m.movieId.isin(r.movieId.unique())].reset_index(drop=True)
    imap = {mid: i for i, mid in enumerate(m.movieId)}
    umap = {uid: i for i, uid in enumerate(sorted(r.userId.unique()))}
    inter = pd.DataFrame({"user_id": r.userId.map(umap), "item_id": r.movieId.map(imap), "ts": r.ts})
    inter = inter.sort_values(["ts", "user_id"]).reset_index(drop=True)
    genres = m.genres.fillna("").str.split("|")
    born = inter.groupby("item_id").ts.min().reindex(range(len(m))).fillna(0).astype(int)
    items = pd.DataFrame({
        "item_id": np.arange(len(m)),
        "title": m.title,
        "category": [g[0] if g and g[0] else "Unknown" for g in genres],
        "tags": [" ".join(x.lower().replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
                          for x in g if x) or "none" for g in genres],
        "born_day": born.values,
    })
    return items, inter


def to_relative_time(items, inter, cold_frac=0.1, seed=7, horizon=100, train_frac=0.85):
    """Per-user temporal split for datasets where users rate in short bursts (MovieLens).

    A global date cutoff leaves almost no users with both past and future activity there. Instead,
    each user's own interactions are mapped onto a shared 0..horizon axis by their position in
    that user's history (oldest -> 0, newest -> horizon). The usual global cutoffs then hold out
    every user's most recent ~30% (15% for ranker training, 15% for testing).

    cold_frac of users are treated as brand-new: all their interactions land in the test window, so
    they have no history except the onboarding context. An item's release day becomes the first
    pseudo-day anyone interacted with it, so items seen only late or only by cold users are "new".
    Timestamps become pseudo-time: order within a user is real, but age is only relative.
    """
    rng = np.random.default_rng(seed)
    inter = inter.sort_values(["user_id", "ts", "item_id"]).reset_index(drop=True)
    users = inter.user_id.unique()
    cold = set(rng.choice(users, size=int(len(users) * cold_frac), replace=False))
    pos = inter.groupby("user_id").cumcount().to_numpy()
    n = inter.groupby("user_id").item_id.transform("size").to_numpy()
    hi = round(train_frac * horizon)
    warm_ts = np.floor(horizon * (pos + 0.5) / n)
    cold_ts = hi + np.floor((horizon - hi) * pos / n)
    ts = np.where(inter.user_id.isin(cold).to_numpy(), cold_ts, warm_ts).astype(int)
    out = inter.assign(ts=ts).sort_values(["ts", "user_id"]).reset_index(drop=True)
    items = items.copy()
    born = out.groupby("item_id").ts.min().reindex(range(len(items))).fillna(0).astype(int)
    items["born_day"] = born.values
    return items, out
