from dataclasses import dataclass, field


@dataclass
class Config:
    # synthetic data
    n_users: int = 3000
    n_items: int = 1500
    horizon: int = 120          # days of simulated history
    seed: int = 7
    # temporal split, as fractions of the horizon
    ranker_frac: float = 0.667  # retrievers see data before this day; ranker learns from the next window
    train_frac: float = 0.833   # final retrievers see data before this day; test starts here
    # cold start
    ctx_len: int = 3            # onboarding interactions given to brand-new users at test time
    fresh_days: int = 14        # items younger than this count as "new"
    # stage 1: candidates pulled from each retriever
    n_cand: dict = field(default_factory=lambda: dict(mf=80, knn=80, content=60, pop=40, fresh=20))
    # stage 2 and re-rank
    mmr_lambda: float = 0.25
    out_len: int = 50
