import pandas as pd
from llm_eval.judge import align_judgments, agreement_metrics, release_gate


def test_alignment_and_agreement():
    human = pd.DataFrame({
        "question_id": [1,1,1,2,2,2],
        "model_a": ["a"]*3 + ["a"]*3,
        "model_b": ["b"]*3 + ["b"]*3,
        "turn": [1]*3 + [1]*3,
        "winner": ["model_a","model_a","model_b","model_b","model_b","model_b"],
    })
    gpt = pd.DataFrame({
        "question_id": [1,2], "model_a": ["a","a"], "model_b": ["b","b"],
        "turn": [1,1], "winner": ["model_a","model_b"]
    })
    aligned = align_judgments(human, gpt)
    m = agreement_metrics(aligned)
    assert len(aligned) == 2
    assert m["agreement"] == 1.0


def test_release_gate_passes_perfect_agreement():
    aligned = pd.DataFrame({
        "question_id": range(120), "model_a": ["a"]*120, "model_b": ["b"]*120,
        "turn": [1]*120, "human_winner": ["model_a"]*60 + ["model_b"]*60,
        "gpt4_winner": ["model_a"]*60 + ["model_b"]*60,
    })
    assert release_gate(aligned)["automated_release_gate"] is True
