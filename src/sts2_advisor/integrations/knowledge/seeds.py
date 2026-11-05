"""Inject reviewed baseline rules independently of optional guide stores."""
from __future__ import annotations

from pathlib import Path
from sts2_advisor.contracts import DecisionContext, Evidence
from sts2_advisor.contracts import Retriever
from sts2_advisor.developer import dev_section
from sts2_advisor.recording.trajectory import record_event
from sts2_advisor.integrations.knowledge.common import parse_evidence_lines, metadata_matches


def load_seeds(path: Path) -> list[Evidence]:
    """读取内置基线规则。seeds.jsonl 只由人工维护，ingest --rebuild 不得触碰。"""
    return parse_evidence_lines(path, "种子规则")


class SeedInjectingRetriever:
    """把内置基线规则前置注入检索结果，不占 top_k 名额。

    语料检索是尽力而为的相似度排名：top-k 窗口可能被无资格块占满，
    语料文件也可能被 rebuild 清空重建，基线规则一旦缺席一次就可能输掉
    整局，因此必须由检索工厂统一保证注入，不依赖语料库状态。
    mode=none 是显式关闭知识库，不包裹、不注入。
    """

    def __init__(self, inner: Retriever, seeds: list[Evidence]) -> None:
        self.inner = inner
        self.seeds = seeds

    def retrieve(self, context: DecisionContext, top_k: int = 6) -> list[Evidence]:
        evidence = self.inner.retrieve(context, top_k)
        present = {item.evidence_id for item in evidence}
        matched = [
            seed for seed in self.seeds
            if seed.evidence_id not in present and metadata_matches(seed.metadata, context)
        ]
        if matched:
            record_event("retrieval.seeds", evidence_ids=[item.evidence_id for item in matched])
            dev_section("检索注入", "内置基线规则", fields={
                "evidence_ids": [item.evidence_id for item in matched],
            })
        return matched + evidence
