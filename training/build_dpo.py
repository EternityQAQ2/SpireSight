from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="从审核偏好对构建 DPO JSONL")
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/dpo/train.jsonl"))
    args = parser.parse_args()

    contexts: dict[str, dict] = {}
    for path in args.recordings:
        for record in read_jsonl(path):
            decision_id = record.get("context", {}).get("decision_id")
            if decision_id:
                contexts[decision_id] = record

    samples = []
    for preference in read_jsonl(args.preferences):
        if preference.get("approved") is not True:
            continue
        decision_id = preference.get("decision_id")
        record = contexts.get(decision_id)
        if not record:
            continue
        context = record["context"]
        candidates = {item["candidate_id"] for item in context.get("candidates", [])}
        chosen = preference.get("chosen", {})
        rejected = preference.get("rejected", {})
        if chosen.get("candidate_id") not in candidates or rejected.get("candidate_id") not in candidates:
            continue
        prompt = json.dumps(
            {"context": context, "evidence": record.get("evidence", [])},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        samples.append(
            {
                "prompt": prompt,
                "chosen": json.dumps(chosen, ensure_ascii=False, separators=(",", ":")),
                "rejected": json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
                "metadata": {
                    "decision_id": decision_id,
                    "preference_reason": preference.get("preference_reason", ""),
                    "reviewer": preference.get("reviewer", "unknown"),
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"DPO samples: {len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

