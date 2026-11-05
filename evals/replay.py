"""Offline fixed-state evaluation. Never imports a provider or calls the game.

Use identical case_ids to compare externally collected model/thinking predictions.
Each predictions JSONL row: {case_id, advice: {...}}. Missing cases fail coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.search import rank_turns
from sts2_advisor.decision.policy import repair_plan_if_current_action_is_legal
from sts2_advisor.contracts import Advice
from sts2_advisor.game.state import build_decision_context


def evaluate(cases: list[dict], predictions: dict[str, dict] | None = None) -> dict:
    rows = []
    for case in cases:
        context = build_decision_context(case["raw_state"], deck_snapshot=case.get("deck_snapshot"))
        if context is None:
            raise ValueError(f"fixture has no decision: {case['case_id']}")
        raw = (predictions or {}).get(case["case_id"]) if predictions is not None else case["historical_advice"]
        if raw is None:
            rows.append({"case_id": case["case_id"], "missing_prediction": True})
            continue
        raw = dict(raw)
        raw["decision_id"] = context.decision_id
        advice = Advice(**{k: v for k, v in raw.items() if k in Advice.__dataclass_fields__})
        ids = advice.plan_candidate_ids or [advice.candidate_id]
        checked = check_plan(context, ids)
        legal_ids = {c.candidate_id for c in context.candidates if c.legal}
        structural_error = ("计划首项不一致或包含非法候选"
                            if ids[0] != advice.candidate_id or any(i not in legal_ids for i in ids) else None)
        repaired, reason = repair_plan_if_current_action_is_legal(advice, context)
        after = check_plan(context, repaired.plan_candidate_ids or [repaired.candidate_id])
        selection = advice.selection_analysis
        selection_ok = context.decision_type != "card_reward" or all(
            isinstance(selection.get(k), str) and selection[k].strip()
            for k in ("gap", "contribution", "skip_comparison"))
        rows.append({"case_id": case["case_id"],
                     "legal": advice.candidate_id in legal_ids,
                     "plan_error": structural_error or checked.error, "observation_boundary": checked.boundary,
                     "boundary_crossed": bool(checked.boundary and checked.candidate_ids != ids),
                     "selection_explained": bool(selection_ok),
                     "repair_reason": reason, "repaired_plan": after.candidate_ids,
                     "repaired_plan_error": after.error, "search": rank_turns(context)})
    scored = [r for r in rows if not r.get("missing_prediction")]
    return {"total": len(rows), "coverage": len(scored) / len(rows) if rows else 0,
            "legal": sum(r["legal"] for r in scored),
            "plan_errors": sum(r["plan_error"] is not None for r in scored),
            "observation_boundaries": sum(r["observation_boundary"] is not None for r in scored),
            "boundary_crossings": sum(r["boundary_crossed"] for r in scored),
            "selection_explanation_errors": sum(not r["selection_explained"] for r in scored),
            "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent / "fixtures/log_cases.jsonl")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.fixtures.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = None
    if args.predictions:
        records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        predictions = {r["case_id"]: r["advice"] for r in records}
        if len(predictions) != len(records):
            parser.error("predictions has duplicate case_id")
        if set(predictions) - {c["case_id"] for c in cases}:
            parser.error("predictions contains unknown case_id")
    result = evaluate(cases, predictions)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if predictions is None or (result["coverage"] == 1 and result["legal"] == result["total"]
                                        and result["plan_errors"] == 0
                                        and result["boundary_crossings"] == 0
                                        and result["selection_explanation_errors"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
