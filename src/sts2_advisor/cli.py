from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .api import Sts2ApiError, Sts2RestClient
from .logging_config import configure_logging, get_logger
from .model import OpenAIAdvisor, RuleAdvisor, TransformersAdvisor
from .recording import SessionRecorder
from .retrieval import create_retriever
from .service import AdvisorService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="《杀戮尖塔2》控制台建议器与自动操作")
    parser.add_argument("--base-url", default="http://localhost:15526")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--watch", action="store_true", help="持续监听游戏决策状态")
    parser.add_argument("--auto-act", action="store_true", help="自动向游戏发送推荐动作；搭配 --watch 持续操作")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stable-reads", type=int, default=2)
    parser.add_argument(
        "--backend", choices=("rule", "transformers", "openai"), default=None,
        help="推理后端；省略时在终端询问本地或远程模式",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="模型推理设备；auto 按 CUDA、Apple MPS、CPU 顺序选择",
    )
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--model-cache", type=Path, default=PROJECT_ROOT / "MiniCPM5_2B")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--remote-max-tokens", type=int, default=None)
    parser.add_argument("--remote-timeout", type=float, default=60.0)
    parser.add_argument("--rag", choices=("auto", "none", "json", "chroma"), default="auto")
    parser.add_argument("--chroma-path", type=Path, default=PROJECT_ROOT / "data/chroma")
    parser.add_argument("--knowledge-file", type=Path, default=PROJECT_ROOT / "data/knowledge.jsonl")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--recording-dir", type=Path, default=PROJECT_ROOT / "recordings")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    parser.add_argument("--log-retention-days", type=int, default=3)
    parser.add_argument("--dump-context", action="store_true", help="打印压缩后的决策上下文并退出")
    return parser


def choose_backend_interactively(input_fn=input, output_fn=print) -> str:
    output_fn("请选择模型模式：")
    output_fn("  1. 本地模型")
    output_fn("  2. 远程 API")
    while True:
        try:
            choice = input_fn("请输入 1/2 [1]：").strip().lower()
        except EOFError:
            return "transformers"
        if choice in {"", "1", "local", "本地"}:
            return "transformers"
        if choice in {"2", "remote", "远程"}:
            return "openai"
        output_fn("输入无效，请输入 1 或 2。")


def load_environment(path: Path) -> None:
    """Load .env without overwriting environment variables from the process."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if (
        args.timeout <= 0 or args.remote_timeout <= 0 or args.poll_interval <= 0
        or args.stable_reads < 1 or args.log_retention_days < 1
        or (args.remote_max_tokens is not None and args.remote_max_tokens < 1)
    ):
        print("超时、轮询间隔、日志保留天数和 token 上限必须为正数", file=sys.stderr)
        return 2

    try:
        log_path = configure_logging(args.log_dir, args.log_retention_days)
    except (OSError, ValueError) as exc:
        print(f"日志初始化失败：{exc}", file=sys.stderr)
        return 1
    logger.info("startup argv=%r", argv if argv is not None else sys.argv[1:])

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

    if args.backend is None:
        # Prompt people at a real terminal; preserve deterministic behavior for
        # subprocesses, schedulers and redirected stdin.
        args.backend = choose_backend_interactively() if sys.stdin.isatty() else "rule"

    print(f"运行日志：{log_path}")

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
    elif args.backend == "openai":
        load_environment(PROJECT_ROOT / ".env")
        try:
            remote_model = os.environ.get("OPENAI_MODEL", "").strip()
            remote_max_tokens = args.remote_max_tokens or int(
                os.environ.get("OPENAI_MAX_TOKENS", "2048")
            )
            backend = OpenAIAdvisor(
                remote_model,
                api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
                base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
                timeout=args.remote_timeout,
                max_tokens=remote_max_tokens,
            )
        except Exception as exc:
            print(f"远程模型配置失败：{exc}", file=sys.stderr)
            return 1
        fallback = RuleAdvisor()
        print(f"远程模型：{remote_model}（最大输出 {remote_max_tokens} tokens）")
    else:
        backend = RuleAdvisor()
        fallback = None
    logger.info("backend_ready backend=%s log_path=%s", args.backend, log_path)

    service = AdvisorService(
        client,
        retriever,
        backend,
        SessionRecorder(args.recording_dir, not args.no_record),
        backend_name=args.backend,
        fallback_backend=fallback,
        top_k=args.top_k,
        auto_act=args.auto_act,
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
