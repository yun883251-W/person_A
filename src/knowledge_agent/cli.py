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
    from .application import KnowledgeAgentApplication
    from .knowledge.ingest import IngestRunResult
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
/clear    永久清空当前会话的全部历史，但保留会话
/refresh  重载已发布的索引；文档变更后请先在另一终端运行 person-a update
/exit     保存已完成的问答并退出
Ctrl+C 或输入结束（EOF）也会保留已经完成的问答。"""


def _nonempty(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("输入不能为空或仅包含空白。")
    return cleaned


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是非负整数。") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数。")
    return parsed


def _add_skill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skill-mode", choices=("off", "auto", "manual"),
        help="普通问答 / 自动选择 / 手动技能；省略时按 --skill 是否提供保持原行为",
    )
    parser.add_argument("--skill", type=_nonempty, help="手动技能 ID；不能与 auto/off 同用")


def build_parser() -> argparse.ArgumentParser:
    """声明命令，但不导入模型或检索依赖。"""
    parser = argparse.ArgumentParser(
        prog="person-a",
        description="基于本地 Markdown 知识库的 Python / LangChain 学习助手。",
        epilog=(
            "配置从环境变量和项目根目录的 .env 读取。会话默认持久化到 "
            "data/sessions/chat.db，可用 SESSION_DB_PATH 和 "
            "SESSION_DB_LOCK_TIMEOUT 配置。"
        ),
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
            _add_skill_arguments(command)
        command.add_argument("--json", action="store_true", help="将结构化结果以 JSON 输出")

    chat = commands.add_parser(
        "chat",
        help="新建或恢复持久化的交互式多轮会话",
        description=(
            "不指定 --session-id 时新建会话；指定后恢复该会话，"
            "可用 person-a sessions 查看可恢复的 ID。"
        ),
    )
    chat.add_argument(
        "--session-id",
        type=_nonempty,
        help="恢复已有会话 ID；省略时创建新的随机 ID",
    )
    _add_skill_arguments(chat)

    sessions = commands.add_parser(
        "sessions", help="列出本地持久化会话（不加载模型或知识库）"
    )
    sessions.add_argument("--limit", type=_positive_int, default=50, help="最多显示的会话数")
    sessions.add_argument("--offset", type=_nonnegative_int, default=0, help="跳过的会话数")
    sessions.add_argument("--json", action="store_true", help="将结构化结果以 JSON 输出")

    history = commands.add_parser(
        "history", help="查看一个会话的完整历史（不加载模型或知识库）"
    )
    history.add_argument("session_id", type=_nonempty, help="会话 ID")
    history.add_argument("--limit", type=_positive_int, help="最多显示的问答轮数")
    history.add_argument("--offset", type=_nonnegative_int, default=0, help="跳过的问答轮数")
    history.add_argument("--json", action="store_true", help="将结构化结果以 JSON 输出")

    delete = commands.add_parser(
        "delete-session", help="永久删除一个会话及其历史（不加载模型或知识库）"
    )
    delete.add_argument("session_id", type=_nonempty, help="会话 ID")
    delete.add_argument("--json", action="store_true", help="将结构化结果以 JSON 输出")

    evaluate = commands.add_parser(
        "evaluate",
        help="执行真实批量问答评估并生成报告",
        description="调用已配置的模型执行真实批量问答，保存来源覆盖统计和待人工复核报告。",
    )
    evaluate.add_argument("--questions", help="JSONL 题集路径；相对路径按项目根目录解析")
    evaluate.add_argument("--output-dir", help="报告父目录；相对路径按项目根目录解析")
    evaluate.add_argument("--limit", type=_positive_int, help="只执行前 N 题")
    evaluate.add_argument("--json", action="store_true", help="将结构化汇总以 JSON 输出")
    return parser


def _write_json(payload: dict[str, Any], *, error: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr if error else sys.stdout)


def _error_details(exc: Exception) -> tuple[str, str]:
    """使用固定的公开消息；异常文本可能包含凭据或 URL。"""
    messages = {
        "McpConfigurationError": ("mcp_configuration_failed", "MCP 配置无效，请检查 MCP_CONFIG_PATH 和对应 JSON 文件。"),
        "McpAdapterError": ("mcp_adapter_failed", "MCP 工具定义无法接入，请检查工具服务和 mcp 可选依赖。"),
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
        "SessionNotFoundError": ("session_not_found", "会话不存在，请先运行 person-a sessions 查看会话 ID。"),
        "DuplicateSessionError": ("session_exists", "会话 ID 已存在，请恢复该会话或使用新的 ID。"),
        "SessionStoreLockedError": ("session_store_busy", "会话数据库正忙，请稍后重试。"),
        "SessionStoreSchemaError": ("session_store_incompatible", "会话数据库版本不兼容。"),
        "SessionStoreCorruptionError": ("session_store_corrupt", "会话数据库已损坏。"),
        "SessionStorePathError": ("session_store_path_failed", "无法访问会话数据库，请检查路径和权限。"),
        "SessionStoreReadError": ("session_store_read_failed", "读取会话数据库失败。"),
        "SessionStoreWriteError": ("session_store_write_failed", "写入会话数据库失败，本次更改未保存。"),
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
    from .knowledge.ingest import build_index, update_index

    operation = build_index if command == "build" else update_index
    print("正在建立索引……" if command == "build" else "正在更新索引……", file=sys.stderr)

    def progress(completed: int, total: int) -> None:
        if sys.stderr.isatty():
            print(f"向量化片段：{completed}/{total}", file=sys.stderr)

    result = operation(progress=progress)
    return _display_ingest(result, json_output=json_output)


def _public_answer_payload(result: AnswerResult) -> dict[str, Any]:
    """使用与 HTTP 相同的公开测评边界序列化 CLI 回答。"""
    from .agent.assessment import public_assessment
    from knowledge_agent.agent.answering.diagnostics import public_answer_result_dict
    from knowledge_agent.agent.planning.planner import validate_plan_summary
    from knowledge_agent.observability.trace import trace_to_dict

    payload = public_answer_result_dict(result)
    payload.pop("assessment_state", None)
    payload["assessment"] = public_assessment(result.assessment_state)
    payload["plan"] = validate_plan_summary(result.plan) if result.plan is not None else None
    payload["trace"] = trace_to_dict(result.trace) if result.trace is not None else None
    return payload


def _display_answer(result: AnswerResult, *, json_output: bool = False) -> int:
    if json_output:
        _write_json(_public_answer_payload(result))
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
        if result.business_state is not None:
            current = result.business_state
            if current.waiting_for_input:
                print(
                    f"当前业务：{current.business_type}；等待补充：{current.clarification_question}",
                    file=sys.stderr,
                )
            elif current.phase in {"cancelled", "execution_unknown"}:
                print(
                    f"当前业务：{current.business_type or '无'}；状态：{current.phase}",
                    file=sys.stderr,
                )
        if result.trace is not None:
            for event in result.trace.events:
                if event.stage == "skill_select":
                    prefix = f"步骤 {event.step_id}；" if event.step_id else ""
                    print(
                        f"{prefix}技能模式：{event.skill_mode}；"
                        f"选择：{event.selected_skill_id or '普通问答'}；"
                        f"策略：{event.strategy_id or 'default'}；原因：{event.reason_code}",
                        file=sys.stderr,
                    )
    return 1 if result.status in {"service_error", "limit_reached"} else 0


def _create_session_store() -> Any:
    """创建存储，但不构造模型或检索服务。"""
    from .config import get_session_store_settings
    from knowledge_agent.session.store import SQLiteSessionStore

    settings = get_session_store_settings()
    return SQLiteSessionStore(settings.db_path, lock_timeout=settings.lock_timeout)


@contextmanager
def _session(
    session_id: str | None,
    *,
    persistent: bool = False,
    resume: bool = False,
) -> Iterator[KnowledgeAgentApplication]:
    from .mcp.bootstrap import create_configured_application

    store = _create_session_store() if persistent else None
    application = create_configured_application(session_store=store)
    try:
        if resume:
            if session_id is None:
                raise ValueError("session_id is required when resuming")
            application.resume_session(session_id)
        else:
            application.start_session(session_id)
        yield application
    finally:
        application.close()


def _skill_options(skill_mode: str | None, skill_id: str | None) -> dict[str, str]:
    options = {}
    if skill_mode is not None:
        options["skill_mode"] = skill_mode
    if skill_id is not None:
        options["skill_id"] = skill_id
    return options


def _run_ask(
    question: str, *, json_output: bool,
    skill_mode: str | None = None, skill_id: str | None = None,
) -> int:
    with _session("cli-ask") as application:
        return _display_answer(
            application.ask(question, **_skill_options(skill_mode, skill_id)),
            json_output=json_output,
        )


def _run_chat(
    session_id: str | None, *, skill_mode: str | None = None, skill_id: str | None = None,
) -> int:
    with _session(
        session_id,
        persistent=True,
        resume=session_id is not None,
    ) as application:
        print(f"会话 ID：{application.session_id}")
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
                    print("已永久清空当前会话历史。")
                elif question == "/refresh":
                    snapshot = application.refresh_index()
                    print(f"已重载索引，版本：{snapshot.index_version}")
                elif question.startswith("/"):
                    print("未知命令，请输入 /help 查看支持的命令。", file=sys.stderr)
                else:
                    _display_answer(application.ask(question, **_skill_options(skill_mode, skill_id)))
            except Exception as exc:
                # 单轮失败不终止交互会话；正常 /exit 或 EOF 仍返回 0。
                _report_error(exc)
                if getattr(exc, "retained_index_version", None) is not None:
                    print("仍保留刷新前的索引，可继续提问。", file=sys.stderr)


@contextmanager
def _storage() -> Iterator[Any]:
    store = _create_session_store()
    try:
        store.initialize()
        yield store
    finally:
        store.close()


def _run_sessions(*, limit: int, offset: int, json_output: bool) -> int:
    with _storage() as store:
        records = store.list_sessions(limit=limit, offset=offset)
    payload = [asdict(record) for record in records]
    if json_output:
        _write_json({"status": "success", "sessions": payload})
    elif not records:
        print("暂无会话。")
    else:
        print("会话 ID\t更新时间\t标题")
        for record in records:
            print(f"{record.session_id}\t{record.updated_at}\t{record.title or '-'}")
    return 0


def _run_history(
    session_id: str, *, limit: int | None, offset: int, json_output: bool
) -> int:
    with _storage() as store:
        turns = store.load_turns(session_id, limit=limit, offset=offset)
    payload = [
        {
            "turn_id": turn.turn_id,
            "turn_no": turn.turn_no,
            "question": turn.question,
            "retrieval_question": turn.retrieval_question,
            "result": _public_answer_payload(turn.result),
        }
        for turn in turns
    ]
    if json_output:
        _write_json({"status": "success", "session_id": session_id, "turns": payload})
    elif not turns:
        print("该会话暂无历史。")
    else:
        for turn in turns:
            print(f"[{turn.turn_no}] 你：{turn.question}")
            print(f"助手：{turn.result.answer}")
    return 0


def _run_delete_session(session_id: str, *, json_output: bool) -> int:
    with _storage() as store:
        deleted = store.delete_session(session_id)
    if not deleted:
        from knowledge_agent.session.store import SessionNotFoundError

        raise SessionNotFoundError("Session does not exist")
    if json_output:
        _write_json({"status": "success", "session_id": session_id, "deleted": True})
    else:
        print(f"已删除会话：{session_id}")
    return 0


def _run_evaluate(
    questions: str | None,
    output_dir: str | None,
    *,
    limit: int | None,
    json_output: bool,
) -> int:
    from .evaluation import run_evaluation

    def progress(completed: int, total: int) -> None:
        print(f"评估进度：{completed}/{total}", file=sys.stderr)

    result = run_evaluation(
        questions_path=questions,
        output_dir=output_dir,
        limit=limit,
        progress=None if json_output else progress,
    )
    summary = result.summary
    payload = {
        "status": summary.run_status,
        "summary": asdict(summary),
        "paths": {
            "run_dir": str(result.run_dir),
            "results": str(result.results_path),
            "summary": str(result.summary_path),
            "report": str(result.report_path),
        },
        "quality_review": "pending",
    }
    if json_output:
        _write_json(payload)
    else:
        print(f"运行状态：{summary.run_status}")
        print(
            f"计划题数：{summary.total_cases}；已结束：{summary.completed_cases}；"
            f"执行错误：{summary.error_cases}"
        )
        print(f"结果明细：{result.results_path}")
        print(f"汇总：{result.summary_path}")
        print(f"可读报告：{result.report_path}")
        print("人工质量评审：待复核（运行完成不代表答案质量通过）")
    if summary.interrupted or summary.run_status == "interrupted":
        return 130
    if (
        summary.run_status != "completed"
        or summary.run_error is not None
        or summary.error_cases > 0
    ):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """分发一个命令；导入本模块永远不会启动应用。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"ask", "chat"}:
        if args.skill_mode == "manual" and args.skill is None:
            parser.error("--skill-mode manual 需要 --skill ID")
        if args.skill_mode in {"off", "auto"} and args.skill is not None:
            parser.error("--skill 不能与 --skill-mode off/auto 同时使用")
    json_output = getattr(args, "json", False)
    try:
        if args.command in {"build", "update"}:
            return _run_ingest(args.command, json_output=json_output)
        if args.command == "ask":
            return _run_ask(args.question, json_output=json_output, skill_mode=args.skill_mode, skill_id=args.skill)
        if args.command == "chat":
            return _run_chat(args.session_id, skill_mode=args.skill_mode, skill_id=args.skill)
        if args.command == "sessions":
            return _run_sessions(limit=args.limit, offset=args.offset, json_output=json_output)
        if args.command == "history":
            return _run_history(
                args.session_id,
                limit=args.limit,
                offset=args.offset,
                json_output=json_output,
            )
        if args.command == "evaluate":
            return _run_evaluate(
                args.questions,
                args.output_dir,
                limit=args.limit,
                json_output=json_output,
            )
        return _run_delete_session(args.session_id, json_output=json_output)
    except KeyboardInterrupt:
        if json_output:
            _write_json({"status": "interrupted", "message": "操作已中断。"}, error=True)
        else:
            print("\n操作已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        _report_error(exc, json_output=json_output)
        return 1


# 后续统一验证 TODO（尚未执行）：
# 01：验证帮助、子命令、空问题、非法参数和退出码；帮助及参数错误不得初始化模型。
# 02：使用假入库接口与假应用验证分发、JSON 输出、返回型失败和错误信息脱敏。
# 03：验证连续追问、/clear、/refresh、未知命令、EOF、Ctrl+C 与所有退出路径的会话清理。
# 04：真实建库后运行 ask/chat，核对引用、文档更新后刷新以及 evaluation 中的验收问题。


if __name__ == "__main__":
    raise SystemExit(main())
