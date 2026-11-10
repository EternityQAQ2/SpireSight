"""Offline SQLite/Chroma/seed comparison against a Git revision; no model API."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import time
import types
from unittest.mock import patch

from evals.compare_prompts import restore
from sts2_advisor.integrations.knowledge.chroma import ChromaRetriever
from sts2_advisor.integrations.knowledge.common import build_query
from sts2_advisor.integrations.knowledge.embeddings import cached_model_path, EMBEDDING_MODEL
from sts2_advisor.integrations.knowledge.seeds import SeedInjectingRetriever, load_seeds
from sts2_advisor.integrations.knowledge.sqlite import SqliteFactsRetriever


def baseline_module(revision: str, filename: str):
    """Load only the two reviewed local source modules needed for a baseline."""
    source = subprocess.check_output([
        'git', '-c', f'safe.directory={Path.cwd().as_posix()}', 'show',
        f'{revision}:src/sts2_advisor/integrations/knowledge/{filename}.py',
    ], text=True, encoding='utf-8')
    module = types.ModuleType(f'baseline_{filename}')
    exec(compile(source, f'{revision}/{filename}.py', 'exec'), module.__dict__)
    return module


def latency(values: list[float]) -> dict:
    return {'mean_ms': round(statistics.mean(values), 3),
            'p50_ms': round(statistics.median(values), 3),
            'p95_ms': round(sorted(values)[int((len(values) - 1) * .95)], 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='932eb58')
    parser.add_argument('--fixtures', type=Path, default=Path('evals/fixtures/prompt_cases.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('tmp/retrieval-relevance.json'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('输出已存在，请指定新文件以保留对照结果')
    if args.repeats <= 0:
        parser.error('repeats必须大于0')
    if cached_model_path(EMBEDDING_MODEL) == EMBEDDING_MODEL:
        parser.error('本地嵌入模型未缓存；不下载模型')
    old_sqlite = baseline_module(args.baseline, 'sqlite').SqliteFactsRetriever
    old_query = baseline_module(args.baseline, 'common').build_query
    started = time.perf_counter()
    vector = ChromaRetriever(Path('data/chroma'))
    startup_ms = (time.perf_counter() - started) * 1000
    seeds = load_seeds(Path('data/seeds.jsonl'))
    retrievers = {
        'before': SeedInjectingRetriever(old_sqlite(Path('data/sts2_facts.sqlite3'), vector), seeds),
        'after': SeedInjectingRetriever(SqliteFactsRetriever(Path('data/sts2_facts.sqlite3'), vector), seeds),
    }
    rows = [json.loads(s) for s in args.fixtures.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:
        parser.error('没有局面')
    contexts = [restore(row)[0] for row in rows]

    def retrieve(version, context):
        with patch('sts2_advisor.integrations.knowledge.chroma.build_query',
                   old_query if version == 'before' else build_query):
            return retrievers[version].retrieve(context, 6)

    # Exclude one-time model/index initialization from warm-query latency.
    for version in retrievers:
        retrieve(version, contexts[0])
    times = {version: [] for version in retrievers}
    cases = []
    for repeat in range(args.repeats):
        for row, context in zip(rows, contexts):
            case = {'case_id': row['case_id'], 'split': row['split']}
            order = ['before', 'after'] if repeat % 2 == 0 else ['after', 'before']
            for version in order:
                start = time.perf_counter()
                evidence = retrieve(version, context)
                times[version].append((time.perf_counter() - start) * 1000)
                if repeat == 0:
                    query = old_query(context) if version == 'before' else build_query(context)
                    sources = Counter('seed' if e.evidence_id.startswith('builtin-') else
                                      'sqlite' if e.evidence_id.startswith('sqlite:') else 'vector'
                                      for e in evidence)
                    case[version] = {
                        'query': query, 'query_chars': len(query), 'sources': dict(sources),
                        'evidence_chars': sum(len(e.text) for e in evidence),
                        'evidence': [e.to_dict() for e in evidence],
                    }
            if repeat == 0:
                cases.append(case)
                print(row['case_id'], case['before']['query_chars'], '->', case['after']['query_chars'], flush=True)
    metrics = {}
    for version in retrievers:
        all_evidence = [e for case in cases for e in case[version]['evidence']]
        metrics[version] = {
            'latency': latency(times[version]), 'samples': len(times[version]),
            'mean_query_chars': round(statistics.mean(c[version]['query_chars'] for c in cases), 1),
            'mean_evidence_chars': round(statistics.mean(c[version]['evidence_chars'] for c in cases), 1),
            'powers_included': sum(e['evidence_id'].startswith('sqlite:powers:') for e in all_evidence),
            'monsters_included': sum(e['evidence_id'].startswith('sqlite:monsters:') for e in all_evidence),
            'card_facts_included': sum(e['evidence_id'].startswith('sqlite:cards:') for e in all_evidence),
            'pickup_only_relics_included': sum(e['evidence_id'].startswith('sqlite:relics:') and
                json.loads(e['text']).get('description', '').startswith('拾起时') for e in all_evidence),
            'vector_guides_included': sum(not e['evidence_id'].startswith(('sqlite:', 'builtin-')) for e in all_evidence),
        }
    report = {'baseline': args.baseline, 'embedding_model': EMBEDDING_MODEL,
              'vector_documents': vector.collection.count(), 'top_k': 6, 'repeats': args.repeats,
              'startup_ms': round(startup_ms, 3), 'metrics': metrics, 'cases': cases,
              'limitation': '内容覆盖和本地检索耗时，不是模型决策准确率；未调用付费API或游戏'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
