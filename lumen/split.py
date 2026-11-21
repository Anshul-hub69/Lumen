"""Temporal split. Nothing from the future ever reaches a retriever or the ranker."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class Case:
    user: int
    history: list
    targets: list
    kind: str  # "warm" (has earlier history) or "cold" (first seen in this window)


def cutoffs(inter, cfg):
    horizon = int(inter.ts.max()) + 1
    return round(cfg.ranker_frac * horizon), round(cfg.train_frac * horizon), horizon


def build_cases(inter: pd.DataFrame, lo: int, hi: int | None, ctx_len: int = 3):
    """One case per user active in [lo, hi). Warm users get their full earlier history.
    Users with no earlier history are cold: their first ctx_len interactions in the window
    act as onboarding context and the rest are the targets."""
    inter = inter.sort_values(["ts", "user_id"])
    past = inter[inter.ts < lo].groupby("user_id").item_id.apply(list).to_dict()
    win = inter[inter.ts >= lo]
    if hi is not None:
        win = win[win.ts < hi]
    cases = []
    for u, items in win.groupby("user_id").item_id.apply(list).items():
        hist, kind = list(past.get(u, [])), "warm"
        if not hist:
            kind, hist, items = "cold", items[:ctx_len], items[ctx_len:]
        seen = set(hist)
        targets = [i for i in dict.fromkeys(items) if i not in seen]
        if targets:
            cases.append(Case(int(u), [int(i) for i in hist], [int(i) for i in targets], kind))
    return cases
