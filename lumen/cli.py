"""python -m lumen.cli {generate|train|evaluate|serve|all}"""
import argparse
import os

from . import data
from .config import Config
from .evaluate import run as run_eval
from .pipeline import TwoStageRecommender


def main():
    ap = argparse.ArgumentParser(prog="lumen")
    ap.add_argument("step", choices=["generate", "train", "evaluate", "serve", "all"])
    ap.add_argument("--movielens", help="folder containing ratings.csv and movies.csv (replaces synthetic data)")
    ap.add_argument("--user-split", action="store_true",
                    help="per-user temporal split (recommended for MovieLens, where users rate in short bursts)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    cfg = Config()
    for d in ("data", "models", "reports"):
        os.makedirs(d, exist_ok=True)

    if a.step in ("generate", "all"):
        items, inter = data.load_movielens(a.movielens) if a.movielens else \
            data.generate(cfg.n_users, cfg.n_items, cfg.horizon, cfg.seed)
        if a.user_split:
            items, inter = data.to_relative_time(items, inter, seed=cfg.seed)
        data.save(items, inter)
        print(f"data: {len(items)} items, {inter.user_id.nunique()} users, {len(inter)} interactions")
    if a.step in ("train", "all"):
        items, inter = data.load()
        TwoStageRecommender(items, cfg).fit(inter).save("models/recommender.pkl")
    if a.step in ("evaluate", "all"):
        items, inter = data.load()
        rec = TwoStageRecommender.load("models/recommender.pkl")
        run_eval(rec, inter, cfg)
        print("wrote reports/metrics.json and reports/metrics.js")
    if a.step == "serve":
        from .serve import serve
        serve(a.port)


if __name__ == "__main__":
    main()
