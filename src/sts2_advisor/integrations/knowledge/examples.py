"""Bounded worked examples, conditioned on observed mechanics rather than names."""
from __future__ import annotations

import json
from pathlib import Path

from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.integrations.knowledge.common import metadata_matches

EXAMPLE_KIND = 'verified_decision_example'
MECHANISMS = {'next_attack_discount', 'current_block', 'vulnerable', 'slippery'}


def matching_mechanisms(context: DecisionContext) -> list[str]:
    if context.decision_type != 'combat_next_action':
        return []
    result = []
    cards = context.player.get('hand') or []
    if any('下一张攻击牌耗能变为0' in str(c.get('description', '')) for c in cards):
        result.append('next_attack_discount')
    if isinstance(context.player.get('block'), int) and context.player['block'] > 0:
        result.append('current_block')
    for enemy in (context.situation.get('battle') or {}).get('enemies', []):
        if not (enemy.get('hp') or 0) > 0:
            continue
        for status in enemy.get('status') or []:
            ident = str(status.get('id', '')).upper().removesuffix('_POWER')
            if ident in {'VULNERABLE', 'SLIPPERY'} and (status.get('amount') or 0) > 0:
                result.append(ident.lower())
    return list(dict.fromkeys(result))


def select_examples(context: DecisionContext, ranked: list[Evidence], top_k: int) -> list[Evidence]:
    """Replace one guide slot; never append examples to the evidence budget."""
    if top_k <= 0:
        return []
    mechanisms = matching_mechanisms(context)
    guides, examples = [], []
    for evidence in ranked:
        metadata = evidence.metadata
        if metadata.get('kind') != EXAMPLE_KIND:
            guides.append(evidence)
        elif (metadata.get('review_status') == 'verified_arithmetic'
              and metadata.get('mechanism') in mechanisms
              and metadata_matches(metadata, context)
              and len(evidence.text.encode('utf-8')) <= 900):
            examples.append(evidence)
    return (examples[:1] + guides)[:min(top_k, len(guides))]


def load_examples(path: Path) -> list[dict]:
    records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    for row in records:
        meta = row['metadata']
        if (meta.get('kind') != EXAMPLE_KIND or meta.get('review_status') != 'verified_arithmetic'
                or meta.get('mechanism') not in MECHANISMS or not meta.get('source_encounter')
                or len(row['text'].encode('utf-8')) > 900 or 'card:' in row['text']):
            raise ValueError(f"无效核实案例：{row.get('evidence_id')}")
    if len({r['evidence_id'] for r in records}) != len(records):
        raise ValueError('案例ID重复')
    return records
