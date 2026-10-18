from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import Sts2ApiError, Sts2RestClient
from .model import RuleAdvisor, TransformersAdvisor
from .recording import SessionRecorder
from .retrieval import create_retriever
from .service import AdvisorService


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="《杀戮尖塔2》只读控制台建议器")
    parser.add_argument("--base-url", default="http://localhost:15526")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true", help="持续监听游戏决策状态")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stable-reads", type=int, default=2)
    parser.add_argument("--backend", choices=("rule", "transformers"), default="rule")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="模型推理设备；auto 按 CUDA、Apple MPS、CPU 顺序选择",
    )
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "MiniCPM5_2B")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--rag", choices=("auto", "none", "json", "chroma"), default="auto")
    parser.add_argument("--chroma-path", type=Path, default=PROJECT_ROOT / "data/chroma")
    parser.add_argument("--knowledge-file", type=Path, default=PROJECT_ROOT / "data/knowledge.jsonl")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--recording-dir", type=Path, default=PROJECT_ROOT / "recordings")
    parser.add_argument("--dump-context", action="store_true", help="打印压缩后的决策上下文并退出")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or args.poll_interval <= 0 or args.stable_reads < 1:
        print("timeout、poll-interval 必须大于 0，stable-reads 必须至少为 1", file=sys.stderr)
        return 2

    client = Sts2RestClient(args.base_url, args.timeout)
    if args.dump_context:
        from .normalize import build_decision_context

        try:
            raw = client.get_singleplayer_state()
        except Sts2ApiError as exc:
            print(f"API 错误：{exc}", file=sys.stderr)
            return 1
        context = build_decision_context(raw)
        print(json.dumps(context.to_dict() if context else None, ensure_ascii=False, indent=2))
        return 0

    retriever = create_retriever(args.rag, args.chroma_path, args.knowledge_file)
    if args.backend == "transformers":
        try:
            backend = TransformersAdvisor(
                args.model, args.model_cache, args.max_new_tokens, device=args.device
            )
        except Exception as exc:
            print(f"模型加载失败：{exc}", file=sys.stderr)
            return 1
        fallback = RuleAdvisor()
        print(f"模型推理设备：{backend.device}")
    else:
        backend = RuleAdvisor()
        fallback = None

    service = AdvisorService(
        client,
        retriever,
        backend,
        SessionRecorder(args.recording_dir, not args.no_record),
        backend_name=args.backend,
        fallback_backend=fallback,
        top_k=args.top_k,
    )
    try:
        if args.watch:
            service.watch(args.poll_interval, args.stable_reads)
            return 0
        service.run_once(force=True)
        return 0
    except Sts2ApiError as exc:
        print(f"API 错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已停止监听。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
