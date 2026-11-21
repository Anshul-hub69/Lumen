"""Ranking metrics for one user: ranked is a list of item ids, rel is a set of relevant ids."""
import math


def recall_at_k(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def hit_at_k(ranked, rel, k):
    return 1.0 if set(ranked[:k]) & rel else 0.0


def ndcg_at_k(ranked, rel, k):
    gain = sum(1.0 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return gain / ideal if ideal else 0.0
