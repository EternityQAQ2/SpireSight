from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="计算建议结果的机械正确性")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    total = legal = exact = 0
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        total += 1
        candidates = {item["candidate_id"] for item in record["context"]["candidates"]}
        candidate_id = record["advice"]["candidate_id"]
        legal += candidate_id in candidates
        expected = record.get("expected_candidate_id")
        exact += expected is not None and candidate_id == expected
    print(json.dumps({"total": total, "legal_rate": legal / total if total else 0, "exact_rate": exact / total if total else None}, ensure_ascii=False, indent=2))
    return 0 if total and legal == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

