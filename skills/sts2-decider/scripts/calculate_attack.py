"""Read-only bridge to the project's existing calculator; no model or game I/O."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-hp', type=int, help='Optional historical prediction check')
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve() / 'src'))
    from sts2_advisor.contracts import Candidate, DecisionContext
    from sts2_advisor.decision.tools import DecisionTools
    path = args.record if args.record.is_absolute() else args.project / args.record
    with path.open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    if not 0 <= args.index < len(rows):
        parser.error('index out of range')
    raw = dict(rows[args.index]['context'])
    raw['candidates'] = [Candidate(**item) for item in raw['candidates']]
    context = DecisionContext(**raw)
    arguments = {'decision_id': context.decision_id, 'candidate_id': args.candidate}
    if args.expected_hp is not None:
        arguments['expected_hp_after'] = args.expected_hp
    result = DecisionTools(context).call('calculate_attack', json.dumps(arguments))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
