from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

KEYS = ["question_id", "model_a", "model_b", "turn"]


def _majority_winner(group: pd.Series) -> str:
    counts = group.value_counts()
    if counts.empty:
        return "tie"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "tie"
    return str(counts.index[0])


def aggregate_human_majority(human: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["winner"])
    missing = required - set(human.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return (
        human.groupby(KEYS, dropna=False)["winner"]
        .agg(_majority_winner)
        .reset_index(name="human_winner")
    )


def prepare_gpt4(gpt4: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["winner"])
    missing = required - set(gpt4.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return gpt4[KEYS + ["winner"]].rename(columns={"winner": "gpt4_winner"})


def align_judgments(human: pd.DataFrame, gpt4: pd.DataFrame) -> pd.DataFrame:
    return aggregate_human_majority(human).merge(prepare_gpt4(gpt4), on=KEYS, how="inner")


def agreement_metrics(aligned: pd.DataFrame) -> dict:
    if aligned.empty:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    y_h = aligned["human_winner"].astype(str)
    y_g = aligned["gpt4_winner"].astype(str)
    return {
        "n": int(len(aligned)),
        "agreement": float((y_h == y_g).mean()),
        "kappa": float(cohen_kappa_score(y_h, y_g)),
    }


def reliability_by_turn(aligned: pd.DataFrame) -> list[dict]:
    rows = []
    for turn, g in aligned.groupby("turn"):
        rows.append({"turn": int(turn), **agreement_metrics(g)})
    return rows


def bootstrap_agreement_ci(aligned: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if aligned.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    n = len(aligned)
    eq = (aligned["human_winner"].astype(str).to_numpy() == aligned["gpt4_winner"].astype(str).to_numpy()).astype(float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(eq[idx].mean())
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def release_gate(aligned: pd.DataFrame, min_agreement: float = 0.80, min_kappa: float = 0.60, min_n: int = 100) -> dict:
    overall = agreement_metrics(aligned)
    turn_rows = reliability_by_turn(aligned)
    failing = [r for r in turn_rows if r["n"] < min_n or r["agreement"] < min_agreement or r["kappa"] < min_kappa]
    return {
        "overall": overall,
        "thresholds": {"min_agreement": min_agreement, "min_kappa": min_kappa, "min_n": min_n},
        "automated_release_gate": len(failing) == 0 and overall["agreement"] >= min_agreement and overall["kappa"] >= min_kappa,
        "human_review_slices": failing,
    }
