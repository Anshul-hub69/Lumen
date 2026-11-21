# Lumen
A two-stage personalized recommendation engine with cold-start support and diversity-aware ranking.

# Lumen — Two-Stage Recommendation System

A machine learning-powered recommendation engine that combines multiple retrieval strategies, gradient-boosted ranking, and diversity-aware re-ranking to deliver personalized recommendations for returning and cold-start users.

## Overview

Lumen is a recommendation system designed to solve two important challenges:

- **Personalization:** Recommending relevant items based on a user's interaction history.
- **Cold Start:** Generating useful recommendations for users with little or no interaction history.

The system uses a two-stage recommendation architecture:

1. **Candidate Generation:** Retrieves potentially relevant items using multiple recommendation strategies.
2. **Ranking:** Uses a Gradient Boosted Decision Tree (GBDT) model to rank candidate items.
3. **Re-ranking:** Applies Maximal Marginal Relevance (MMR) to improve recommendation diversity.

## Dashboard

The project includes an interactive dashboard for exploring model performance, recommendation quality, ranking metrics, and pipeline health.

Add your dashboard screenshot here:

`![Lumen Dashboard](docs/dashboard.png)`

## Architecture

```text
              User Interaction History
                        |
                        v
               Candidate Generation
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Item kNN     Matrix Factor.  Content-Based
          |             |             |
          +-------------+-------------+
                        |
                        v
                 Rank Fusion (RRF)
                        |
                        v
              Candidate Consolidation
                        |
                        v
                GBDT Ranking Model
                        |
                        v
                  MMR Re-ranking
                        |
                        v
                Top-K Recommendations
```

## Recommendation Algorithms

| Algorithm | Purpose |
|---|---|
| Popularity | Recommends frequently interacted items |
| Item kNN | Finds similar items using interaction patterns |
| Matrix Factorization | Learns latent user-item representations |
| Content-Based | Uses item attributes and categories |
| Reciprocal Rank Fusion | Combines candidate rankings |
| GBDT Ranker | Predicts candidate relevance |
| MMR | Improves recommendation diversity |

## Dataset

The system was evaluated using the MovieLens Latest Small dataset.

| Dataset property | Value |
|---|---:|
| Users | 609 |
| Movies | 6,298 |
| Ratings / interactions | 48,580 |

The evaluation uses a per-user split:

- Returning users have historical interactions and held-out interactions.
- Cold-start users are held out from training and receive three onboarding interactions.
- The evaluation window covers the held-out period.

## Real-Data Evaluation Results

Results below are from the MovieLens run using the updated user-split pipeline.

### Top-10 Model Comparison

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 3.4% | 4.2% |
| Item kNN | 6.0% | 5.7% |
| Matrix Factorization | 2.7% | 2.7% |
| Content-Based | 1.1% | 1.6% |
| Retrieval Only (RRF) | 6.7% | 6.4% |
| Two-Stage GBDT | 6.6% | 6.7% |
| Two-Stage + MMR | **6.8%** | **6.8%** |

### Key Findings

- Two-stage GBDT achieves a 33.0% hit rate.
- Two-stage + MMR achieves the highest Recall@10 and NDCG@10.
- Retrieval-only RRF achieves 6.7% Recall@10.
- Content-based recommendations achieve 0.3% new-item recall.
- The measured ranking latency is 12.2 ms at p50 and 15.9 ms at p95.

### Evaluation Limitations

- The dataset is relatively small for training a complex ranking model.
- Retrieval quality limits the maximum performance of the second-stage ranker.
- The GBDT model does not outperform every baseline on every metric.
- New-item recall remains low, indicating room for improvement in cold-start item discovery.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, SciPy
- **Model Persistence:** Joblib
- **Backend:** Python HTTP server
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** MovieLens Latest Small

## Project Structure

```text
lumen-recsys/
│
├── dashboard/
│   └── index.html
│
├── data/
│   ├── items.csv
│   └── interactions.csv
│
├── lumen/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── retrieval.py
│   ├── serve.py
│   └── split.py
│
├── models/
│   └── recommender.pkl
│
├── reports/
│   ├── metrics.json
│   └── metrics.js
│
├── tests/
├── requirements.txt
├── README.md
└── Makefile
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd lumen-recsys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download MovieLens

Download MovieLens Latest Small from:

https://grouplens.org/datasets/movielens/latest/

Extract the dataset.

### 4. Generate data

```bash
python -m lumen.cli generate --movielens "PATH_TO_ML-LATEST-SMALL" --user-split
```

### 5. Train the model

```bash
python -m lumen.cli train
```

### 6. Evaluate models

```bash
python -m lumen.cli evaluate
```

### 7. Start the application

```bash
python -m lumen.cli serve
```

Open:

http://localhost:8000

## API Usage

Lumen exposes a recommendation endpoint:

`POST /recommend`

Example request:

```json
{
  "history": [12, 480, 903],
  "k": 10
}
```

Example response structure:

```json
{
  "mode": "personalized",
  "items": [
    {
      "item_id": 793,
      "title": "Example Movie",
      "category": "Thriller",
      "reason": "Matches your interest in Thriller"
    }
  ]
}
```

For cold-start recommendations, categories can be supplied:

```json
{
  "history": [],
  "categories": ["Sci-Fi"],
  "k": 5
}
```

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The updated project passed all 11 tests in the development environment.

## Future Improvements

- Experiment with additional candidate retrieval strategies.
- Improve cold-start item recommendations.
- Explore approximate nearest-neighbor indexing for scalability.
- Improve ranking performance with additional user and item features.
- Evaluate on larger recommendation datasets.
- Add online feedback and recommendation tracking.

## Author

Developed as a machine learning and recommendation systems portfolio project.

**Lumen — Intelligent Recommendations Through Multi-Stage Ranking.**
