from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset

from llm_eval.arena import preference_summary, bradley_terry_scores, position_bias, verbosity_bias
from llm_eval.judge import (
    align_annotation_level,
    align_majority_level,
    agreement_metrics,
    agreement_without_ties,
    reliability_by_turn,
    bootstrap_agreement_ci,
    release_gate,
)

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def pct(x):
    return f"{100*x:.2f}%"


def main():
    arena = load_dataset("lmarena-ai/arena-human-preference-55k", split="train").to_pandas()
    pref = preference_summary(arena)
    bt = bradley_terry_scores(arena, min_battles=100)
    pos = position_bias(arena)
    verb = verbosity_bias(arena)

    human = load_dataset("lmsys/mt_bench_human_judgments", split="human").to_pandas()
    gpt4 = load_dataset("lmsys/mt_bench_human_judgments", split="gpt4_pair").to_pandas()

    aligned = align_annotation_level(human, gpt4)
    majority = align_majority_level(human, gpt4)
    judge = agreement_metrics(aligned)
    judge_no_ties = agreement_without_ties(aligned)
    majority_metrics = agreement_metrics(majority)
    judge["agreement_ci_95"] = list(bootstrap_agreement_ci(aligned))
    by_turn = reliability_by_turn(aligned)
    gate = release_gate(aligned)

    pref.to_csv(OUT / "arena_preference_summary.csv", index=False)
    bt.to_csv(OUT / "arena_bradley_terry.csv", index=False)
    aligned.to_csv(OUT / "mtbench_annotation_aligned.csv", index=False)
    majority.to_csv(OUT / "mtbench_majority_aligned.csv", index=False)

    metrics = {
        "arena": {
            "rows": int(len(arena)),
            "unique_models": int(len(set(arena["model_a"]).union(arena["model_b"]))),
            "position_bias": pos,
            "verbosity_bias": verb,
            "top_models_by_tie_adjusted_win_rate": pref.head(10).to_dict(orient="records"),
            "top_models_by_bradley_terry": bt.head(10).to_dict(orient="records"),
        },
        "judge_reliability": {
            "human_annotations": int(len(human)),
            "gpt4_pair_judgments": int(len(gpt4)),
            "aligned_annotation_comparisons": int(len(aligned)),
            "aligned_majority_comparisons": int(len(majority)),
            **judge,
            "without_ties": judge_no_ties,
            "majority_human_vs_gpt4": majority_metrics,
            "by_turn": by_turn,
            "release_gate": gate,
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ci = judge["agreement_ci_95"]
    top_model = pref.iloc[0] if not pref.empty else None
    top_bt = bt.iloc[0] if not bt.empty else None
    md = [
        "# Verified Full-Data Results",
        "",
        "## LMArena human preference outcomes",
        "",
        f"- Battles analyzed: **{len(arena):,}**",
        f"- Unique models: **{metrics['arena']['unique_models']}**",
        f"- Non-tie model-A win rate: **{pct(pos['model_a_win_rate'])}**",
        f"- Longer-response win rate among non-tied unequal-length responses: **{pct(verb['longer_response_win_rate'])}**" if verb['n'] else "- Verbosity analysis unavailable in source schema.",
    ]
    if top_model is not None:
        md += [f"- Highest observed tie-adjusted win rate: **{top_model['model']} ({pct(top_model['tie_adjusted_win_rate'])}, n={int(top_model['battles']):,})**"]
    if top_bt is not None:
        md += [f"- Highest Bradley-Terry score after requiring 100+ battles: **{top_bt['model']}**"]

    md += [
        "",
        "## LLM-as-a-Judge reliability — MT-Bench",
        "",
        f"- Expert human annotations: **{len(human):,}**",
        f"- GPT-4 pairwise judgments: **{len(gpt4):,}**",
        f"- Aligned human-annotation vs GPT-4 comparisons: **{len(aligned):,}**",
        f"- Exact annotation-level agreement: **{pct(judge['agreement'])}** (95% bootstrap CI **{pct(ci[0])}–{pct(ci[1])}**) ",
        f"- Agreement excluding ties: **{pct(judge_no_ties['agreement'])}**",
        f"- Cohen's kappa: **{judge['kappa']:.3f}**",
        f"- Majority-human vs GPT-4 agreement: **{pct(majority_metrics['agreement'])}**",
        "",
        "### Reliability by conversation turn",
        "",
    ]
    for r in by_turn:
        md += [f"- Turn {r['turn']}: **{pct(r['agreement'])} agreement**, κ={r['kappa']:.3f}, n={r['n']:,}"]

    md += [
        "",
        "## Release decision",
        "",
        f"Automated release gate: **{'PASS' if gate['automated_release_gate'] else 'HUMAN REVIEW REQUIRED'}**",
        "",
        "The gate requires agreement, chance-adjusted reliability, and minimum sample size. Failing slices are routed to expert review.",
        "",
        "## Business takeaways",
        "",
        "1. Human preference is the product outcome; benchmark/judge scores are measurement tools.",
        "2. Pair orientation must be canonicalized before comparing evaluators; evaluation pipelines can create false disagreement if measurement logic is wrong.",
        "3. Automated judges need human validation, uncertainty estimates, and slice checks before they can approve model releases.",
        "4. Position and verbosity effects are measurement risks and should be audited before interpreting rankings.",
        "5. Hybrid evaluation is safer than blind automation: automate trusted slices and route weak/uncertain slices to humans.",
    ]
    Path("RESULTS.md").write_text("\n".join(md))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
