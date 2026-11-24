"""Offline coverage audit of recorded plans. Never imports a model/game client."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from sts2_advisor.contracts import Candidate, DecisionContext
from sts2_advisor.decision.plan_check import check_plan
from sts2_advisor.game.combat.forecast import check_end_turn


def audit_row(row: dict) -> dict | None:
    raw = row.get('context')
    advice = row.get('advice')
    if not raw or not advice or raw.get('decision_type') != 'combat_next_action':
        return None
    data = {k: v for k, v in raw.items() if k in DecisionContext.__dataclass_fields__}
    data['candidates'] = [Candidate(**c) for c in data['candidates']]
    context = DecisionContext(**data)
    ids = advice.get('plan_candidate_ids') or [advice.get('candidate_id')]
    result = check_end_turn(context, ids)
    runtime_error = None
    try:
        checked = check_plan(context, ids)
        runtime_error = checked.error
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        runtime_error = f'既有资源检查无法解析：{type(exc).__name__}'
    return {'decision_id': context.decision_id, 'plan': ids, 'forecast': result,
            'runtime_redecision': bool(runtime_error and '结束回合预计生命' in runtime_error),
            'validation_error': runtime_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=Path('recordings'))
    parser.add_argument('--output', type=Path, default=Path('tmp/end-turn-audit.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出已存在，请使用新路径')
    started = time.perf_counter()
    rows, skipped = [], []
    files = sorted(args.recordings.glob('*.jsonl'))
    for file in files:
        for index, line in enumerate(file.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip():
                continue
            try:
                result = audit_row(json.loads(line))
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                skipped.append({'source': str(file), 'index': index, 'error': type(exc).__name__})
                continue
            if result is not None:
                rows.append({'source': str(file), 'index': index, **result})
    statuses = Counter(r['forecast']['status'] for r in rows)
    reasons = Counter(r['forecast'].get('reason', '') for r in rows if r['forecast']['status'] == 'unknown')
    summary = {'files': len(files), 'combat_plans': len(rows), 'statuses': dict(statuses),
               'unknown_reasons': dict(reasons), 'unparsed_rows': len(skipped),
               'runtime_redecisions': sum(r['runtime_redecision'] for r in rows),
               'seconds': round(time.perf_counter() - started, 3), 'api_tokens': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'summary': summary, 'rows': rows, 'unparsed': skipped},
                                     ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
