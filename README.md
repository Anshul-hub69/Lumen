# Lumen: personalized recommendation and ranking engine

A complete, runnable two-stage recommender (candidate retrieval, then learned ranking, then diversity
re-ranking) with cold-start handling for new users and new items, an offline evaluation harness, an HTTP API,
and an evaluation dashboard. It uses only numpy, pandas, scipy and scikit-learn, and ships with a synthetic
data generator so it runs with no downloads. You can point it at MovieLens in one flag.

```
 request: history item ids  +  optional onboarding categories
        |
        v
 STAGE 1  RETRIEVAL  (cheap, wide: ~203 of 1,500 items)
   matrix factorisation | item kNN | content (TF-IDF) | popularity | fresh items
        |  union + dedupe
        v
 STAGE 2  RANKING   gradient-boosted trees on 18 features per (user, item)
        |
        v
 RE-RANK  MMR diversity  ->  top-K with a reason for each item
```

## Step by step

### Step 1. Install
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2. Run everything once
```bash
python -m lumen.cli all        # generate data -> train -> evaluate (about 1-2 minutes)
python -m unittest discover -s tests -t . -v           # 10 tests
```
Or run the stages separately: `python -m lumen.cli generate`, then `train`, then `evaluate`.

### Step 3. Open the dashboard
```bash
python -m lumen.cli serve --port 8000     # then open http://localhost:8000
```
(`dashboard/index.html` also opens straight from disk, because it reads `reports/metrics.js`.)

### Step 4. Ask for recommendations
```bash
# brand-new user who only picked a category
curl -s -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"history": [], "categories": ["Sci-Fi"], "k": 5}'

# returning user described by item ids they interacted with
curl -s -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"history": [12, 480, 903], "k": 10}'
```
The response has `mode` (`cold_start` below 3 interactions, otherwise `personalized`) and items, each with a reason.
Also available: `GET /health`, `GET /categories`, `GET /search?q=title`.

### Step 5. Read the code in pipeline order
| File | What it does |
|---|---|
| `lumen/data.py` | Synthetic catalog and interaction log (latent-factor simulator), plus `load_movielens` |
| `lumen/split.py` | Temporal split and the definition of warm and cold evaluation cases |
| `lumen/retrieval.py` | `ItemSpace`: popularity, item kNN, SVD embeddings and TF-IDF content vectors, all queryable from a list of history ids |
| `lumen/pipeline.py` | Candidate generation, 18-feature builder, GBDT ranker, MMR, `recommend()` |
| `lumen/evaluate.py` | Recall, NDCG, hit rate, coverage, novelty and new-item recall for 7 models, by segment |
| `lumen/serve.py` | Stdlib HTTP API and dashboard host |
| `tests/test_lumen.py` | Metrics, leakage, cold-start paths, persistence, live API |

### Step 6. Use real data (MovieLens)
Download `ml-latest-small` from grouplens.org, unzip it, then (quote the path if it has spaces):
```bash
python -m lumen.cli generate --movielens path/to/ml-latest-small --user-split
python -m lumen.cli train
python -m lumen.cli evaluate
```
**Always add `--user-split` for MovieLens.** Its users rate in short bursts, so a global date cutoff leaves
almost no returning users to test on (a run without the flag had only 12). `--user-split` holds out each user's own
most recent ratings instead, and treats 10% of users as brand-new. Timestamps become pseudo-time: order within
a user is real, but "item age" is only relative. Ratings of 4 or more become positives.
For catalogs beyond about 20k items, replace the dense content matrix `ItemSpace.X` with a sparse one or an ANN
index (FAISS or ScaNN) for the MF and content retrievers.

### Step 7. Extend it
* **Add a retriever:** compute a score vector in `ItemSpace.scores`, add its name to `n_cand`, and add its
  flag and rank columns to `FEATS` and `_build`. The ranker will learn how much to trust it.
* **Add a feature** (price, context, time of day): append a column in `_build` and a name in `FEATS`.
* **Tune diversity:** `Config.mmr_lambda` (0 = pure relevance).
* **Production:** precompute and cache item vectors, run retrieval through an ANN index, log impressions and
  clicks for the next ranker training run, and add an online A/B test before launch.

## How the hard parts work

**No leakage.** The split is temporal. Retrievers are fitted on days before the ranker cutoff, and the ranker learns
from the following window using candidates those retrievers produce. For testing, retrievers are refitted on
everything before the test window, and the test window is only used for scoring. `tests/test_lumen.py` checks it.

**Cold-start users.** Retrievers need no history to work: content and popularity use onboarding category picks
and recent trends. The ranker has `hist_len` as a feature and is trained on artificially truncated histories
(some with simulated category picks), so it learns to trust popularity and content when history is short
and behavior signals when it is long. A unit test checks that a Sci-Fi pick returns mostly Sci-Fi.

**Cold-start items.** New items have no interactions, so MF and kNN cannot see them. The content retriever and a
dedicated fresh-items retriever do, and `age` and `is_new` are ranker features.

**Evaluation protocol.** Test window is days 100-120. Warm users (2,561) are scored on
everything they did in the window. Cold users (300) first appear in the window; their first
3 interactions are context and the rest are targets. Items already in the history are never recommended.
Items released in the test window count as the new-item set.

## Results on the bundled synthetic data (top 10)

| Model | Warm recall | Warm NDCG | Cold recall | Cold NDCG | New-item recall (warm) |
|---|---|---|---|---|---|
| Popularity | 3.7% | 3.0% | 5.1% | 6.8% | 0.0% |
| Item kNN | 7.2% | 6.2% | 6.0% | 8.4% | 0.0% |
| Matrix factorisation | 6.5% | 4.9% | 6.5% | 8.5% | 0.0% |
| Content-based | 2.9% | 2.3% | 3.0% | 3.8% | 2.1% |
| Retrieval only (RRF) | 8.6% | 7.1% | 8.5% | 12.0% | 0.0% |
| Two-stage (GBDT ranker) | 11.0% | 8.2% | 7.9% | 11.0% | 24.3% |
| Two-stage + MMR | 10.0% | 7.7% | 7.6% | 10.9% | 21.0% |

* Retrieval ceiling: Stage 1 contains 47% of the items users went on to choose, from about 203 candidates per request.
* Serving latency: p50 4.8 ms, p95 6.4 ms per request (single thread, CPU).

How to read this honestly:
* The learned ranker is the best model for warm users and the only one that surfaces new items.
* For cold users the learned ranker and the simple rank-fusion baseline are within noise of each other (only 300 users).
* Diversity re-ranking costs a little accuracy, as expected.
* The data is synthetic, so absolute numbers are illustrative. Run Step 6 for meaningful figures.

## Known limitations
* Evaluation uses a single static snapshot at the start of the test window, not a request-by-request replay.
* Offline metrics do not capture position bias or feedback loops. Validate with an online A/B test.
* Implicit positives only. Negative signals (skips, dislikes) are not modeled.
