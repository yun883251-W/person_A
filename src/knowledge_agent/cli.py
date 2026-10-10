"""命令行入口：建库、更新索引、单次提问与交互式多轮问答。

参数解析与结果展示留在本模块；检索、模型调用和会话管理复用业务接口。
先解析命令，再加载对应模块，查看帮助时不读取配置或初始化模型。
功能验证与真实模型联调留到开发完成后统一进行。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import KnowledgeAgentApplication
    from .ingest import IngestRunResult
    from .schemas import AnswerResult


_STATUS_LABELS = {
    "success": "完成",
    "insufficient_evidence": "资料不足",
    "needs_clarification": "需要澄清",
    "out_of_scope": "超出知识库问答范围",
    "service_error": "服务错误",
    "limit_reached": "达到本轮处理上限",
}

_CHAT_HELP = """直接输入问题即可提问，支持连续追问。
/help     显示帮助
/clear    清空本次会话历史
/refresh  重载已发布的索引；文档变更后请先在另一终端运行 person-a update
/exit     结束会话并退出
会话仅保存在当前进程中。Ctrl+C 或输入结束（EOF）也可退出。"""


def _nonempty(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("输入不能为空或仅包含空白。")
    return cleaned


def build_parser() -> argparse.ArgumentParser:
    """Declare commands without importing model or retrieval dependencies."""
    parser = argparse.ArgumentParser(
        prog="person-a",
        description="基于本地 Markdown 知识库的 Python / LangChain 学习助手。",
        epilog="配置从环境变量和项目根目录的 .env 读取。",
    )
    commands = parser.add_subparsers(dest="command", required=True, title="命令")
    for name, description in (
        ("build", "完整建立索引，替换已有索引"),
        ("update", "同步文档增删改，必要时按索引配置重新建库"),
        ("ask", "提出一个问题并退出"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        if name == "ask":
            command.add_argument("question", type=_nonempty, help="问题；包含空格时请用引号包围")
        command.add_argument("--json", action="store_true", help="将结构化结果以 JSON 输出")

    chat = commands.add_parser("chat", help="开始交互式多轮问答")
    chat.add_argument(
        "--session-id", type=_nonempty, default="cli",
        help="本进程内的会话标识（默认 cli），不会恢复上次运行的历史",
    )
    return parser


def _write_json(payload: dict[str, Any], *, error: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr if error else sys.stdout)


def _error_details(exc: Exception) -> tuple[str, str]:
    """Use fixed public messages; exception text may contain credentials or URLs."""
    messages = {
        "IndexNotBuiltError": ("index_not_built", "知识库索引尚未建立，请先运行 person-a build。"),
        "IndexRefreshError": ("index_refresh_failed", "索引刷新失败，请检查索引文件及 Embedding 配置。"),
        "EmbeddingCompatibilityError": ("embedding_incompatible", "向量模型与索引不兼容，请核对 Embedding 配置并重新建库。"),
        "OllamaEmbeddingError": ("embedding_failed", "无法使用 Ollama 向量模型，请检查服务、已下载模型和 Embedding 配置。"),
        "ChatModelAuthenticationError": ("authentication_failed", "对话服务认证失败，请检查 CHAT_API_KEY。"),
        "ChatModelNotFoundError": ("chat_model_not_found", "对话模型不可用，请检查 CHAT_MODEL 和 CHAT_BASE_URL。"),
        "ChatModelTimeoutError": ("chat_timeout", "对话请求超时，请稍后重试或调整 CHAT_REQUEST_TIMEOUT。"),
        "ChatModelServiceError": ("chat_service_failed", "对话服务暂时不可用，请稍后重试。"),
        "ChatModelRequestError": ("chat_request_failed", "对话请求失败，请检查模型配置及工具调用支持。"),
        "ChatModelInvocationError": ("chat_failed", "对话模型调用失败，请检查服务与配置。"),
        "ApplicationBusyError": ("application_busy", "当前操作尚未结束，请等待后再操作。"),
        "SessionNotStartedError": ("session_not_started", "会话尚未开始，请重新启动问答命令。"),
    }
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImportError):
            return "missing_dependency", "缺少项目依赖，请在项目根目录运行 python -m pip install -e .。"
        for error_type in type(current).__mro__:
            details = messages.get(error_type.__name__)
            if details is not None:
                return details
        current = current.__cause__
    if isinstance(exc, OSError):
        return "io_failed", "文件或服务访问失败，请检查目录、读写权限和服务连接。"
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_configuration_or_data", "配置或索引数据无效，请检查 .env 中的必填项、数值和索引文件。"
    return "operation_failed", "操作未完成，请检查配置、依赖和索引后重试。"


def _report_error(exc: Exception, *, json_output: bool = False) -> None:
    code, message = _error_details(exc)
    if json_output:
        _write_json({"status": "error", "error": {"code": code, "message": message}}, error=True)
    else:
        print(f"错误：{message}", file=sys.stderr)


def _display_ingest(result: IngestRunResult, *, json_output: bool) -> int:
    # failure.error 可能包含底层请求信息；终端只展示失败文件与固定原因。
    payload = asdict(result)
    payload["status"] = "success" if result.succeeded else "error"
    payload["output_paths"] = [str(path) for path in result.output_paths]
    payload["failures"] = [
        {"source_path": failure.source_path, "error": "文档读取失败"}
        for failure in result.failures
    ]
    if json_output:
        _write_json(payload)
    else:
        operation = "建库" if result.operation == "build" else "更新索引"
        print(f"{operation}{'完成' if result.succeeded else '失败，未发布本次索引'}。")
        print(f"发现文件：{result.discovered_files}；已读取：{result.read_files}；片段：{result.chunk_count}")
        print(
            f"变化文件：{result.changed_file_count}；新向量：{result.embedded_chunks}；"
            f"复用片段：{result.reused_chunks}；移除片段：{result.removed_chunks}"
        )
        if result.succeeded:
            print(f"处理方式：{'完整重建' if result.full_rebuild else '增量更新'}")
        if result.output_path is not None:
            print(f"索引目录：{result.output_path}")
        if result.failures:
            print(f"读取失败的文件：{len(result.failures)} 个", file=sys.stderr)
            for failure in result.failures:
                print(f"  {failure.source_path}", file=sys.stderr)
    return 0 if result.succeeded else 1


def _run_ingest(command: str, *, json_output: bool) -> int:
    from .ingest import build_index, update_index

    operation = build_index if command == "build" else update_index
    print("正在建立索引……" if command == "build" else "正在更新索引……", file=sys.stderr)

    def progress(completed: int, total: int) -> None:
        if sys.stderr.isatty():
            print(f"向量化片段：{completed}/{total}", file=sys.stderr)

    result = operation(progress=progress)
    return _display_ingest(result, json_output=json_output)


def _display_answer(result: AnswerResult, *, json_output: bool = False) -> int:
    if json_output:
        _write_json(asdict(result))
    else:
        # Agent 已把 [S1] 等标记转换为引用，CLI 原样展示，避免重新编号或编造来源。
        print(result.answer)
        label = _STATUS_LABELS.get(result.status, result.status)
        print(
            f"状态：{label}；引用：{len(result.sources)}；"
            f"模型调用：{result.model_calls}；工具调用：{result.tool_calls}",
            file=sys.stderr,
        )
        if result.stop_reason == "index_not_built":
            print("请先运行 person-a build。", file=sys.stderr)
    return 1 if result.status in {"service_error", "limit_reached"} else 0


@contextmanager
def _session(session_id: str) -> Iterator[KnowledgeAgentApplication]:
    from .agent import create_agent_application

    application = create_agent_application()
    application.start_session(session_id)
    try:
        yield application
    finally:
        if application.session_id is not None:
            application.end_session()


def _run_ask(question: str, *, json_output: bool) -> int:
    with _session("cli-ask") as application:
        return _display_answer(application.ask(question), json_output=json_output)


def _run_chat(session_id: str) -> int:
    with _session(session_id) as application:
        print(_CHAT_HELP)
        while True:
            try:
                question = input("你： " if sys.stdin.isatty() else "").strip()
            except EOFError:
                print("会话已结束。")
                return 0
            if not question:
                continue
            if question == "/exit":
                print("会话已结束。")
                return 0
            if question == "/help":
                print(_CHAT_HELP)
                continue
            try:
                if question == "/clear":
                    application.clear_session()
                    print("已清空本次会话历史。")
                elif question == "/refresh":
                    snapshot = application.refresh_index()
                    print(f"已重载索引，版本：{snapshot.index_version}")
                elif question.startswith("/"):
                    print("未知命令，请输入 /help 查看支持的命令。", file=sys.stderr)
                else:
                    _display_answer(application.ask(question))
            except Exception as exc:
                # 单轮失败不终止交互会话；正常 /exit 或 EOF 仍返回 0。
                _report_error(exc)
                if getattr(exc, "retained_index_version", None) is not None:
                    print("仍保留刷新前的索引，可继续提问。", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one command; importing this module never starts an application."""
    args = build_parser().parse_args(argv)
    json_output = getattr(args, "json", False)
    try:
        if args.command in {"build", "update"}:
            return _run_ingest(args.command, json_output=json_output)
        if args.command == "ask":
            return _run_ask(args.question, json_output=json_output)
        return _run_chat(args.session_id)
    except KeyboardInterrupt:
        if json_output:
            _write_json({"status": "interrupted", "message": "操作已中断。"}, error=True)
        else:
            print("\n操作已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        _report_error(exc, json_output=json_output)
        return 1


# 后续统一验证 TODO（本阶段不执行）：
# 01：验证帮助、子命令、空问题、非法参数和退出码；帮助及参数错误不得初始化模型。
# 02：使用假入库接口与假应用验证分发、JSON 输出、返回型失败和错误信息脱敏。
# 03：验证连续追问、/clear、/refresh、未知命令、EOF、Ctrl+C 与所有退出路径的会话清理。
# 04：真实建库后运行 ask/chat，核对引用、文档更新后刷新以及 evaluation 中的验收问题。


if __name__ == "__main__":
    raise SystemExit(main())
