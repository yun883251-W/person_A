"""将领域对象转换成公开 HTTP 数据，保留来源与可公开的执行摘要。"""

from __future__ import annotations

import logging
from typing import Any

from knowledge_agent.observability.trace import TurnTrace, trace_to_dict
from knowledge_agent.agent.answering.diagnostics import public_answer_diagnostics
from ..knowledge.manager import DocumentSummary, IndexOperationResult, IndexStatus
from ..schemas import AnswerResult, AnswerSource
from knowledge_agent.session.store import SessionRecord, StoredTurn
from .schemas import (
    AnswerResponse,
    AnswerSourceResponse,
    AssessmentResponse,
    BusinessStateResponse,
    DialogueSummaryResponse,
    DocumentChangesResponse,
    DocumentResponse,
    IndexOperationResponse,
    IndexStatusResponse,
    PlanSummaryResponse,
    RouteSummaryResponse,
    SessionResponse,
    TurnResponse,
    TurnTraceResponse,
)

_LOGGER = logging.getLogger(__name__)

def _source_response(source: AnswerSource) -> AnswerSourceResponse:
    """只复制属于公开 API 的来源元数据。"""
    return AnswerSourceResponse(
        chunk_id=source.chunk_id,
        source_path=source.source_path,
        heading_path=list(source.heading_path),
        start_line=source.start_line,
        end_line=source.end_line,
        content_hash=source.content_hash,
        index_version=source.index_version,
        retrieval_score=source.retrieval_score,
    )


def _session_response(record: SessionRecord) -> SessionResponse:
    """将已持久化的会话元数据转换为稳定的公开结构。"""
    return SessionResponse(
        session_id=record.session_id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _trace_response(trace: TurnTrace | None) -> TurnTraceResponse | None:
    """按 trace 白名单提取；格式异常的可选摘要不能导致回答被隐藏。"""
    if trace is None:
        return None
    try:
        return TurnTraceResponse.model_validate(trace_to_dict(trace))
    except (TypeError, ValueError, AttributeError):
        # 此转换没有业务副作用。不要记录 trace 内容，
        # 不要重试本轮，也不要把可用回答变成 HTTP 失败。
        _LOGGER.warning("trace_response_unavailable")
        return None


def _assessment_response(
    snapshot: dict[str, Any] | None,
) -> AssessmentResponse | None:
    """只暴露测评白名单字段，绝不暴露已存储的答案载荷。"""
    if snapshot is None:
        return None
    from ..agent.assessment import public_assessment

    public = public_assessment(snapshot)
    if public is None:
        return None
    return AssessmentResponse.model_validate(public)


def _plan_response(snapshot: dict[str, Any] | None) -> PlanSummaryResponse | None:
    """校验并暴露计划中供回答与历史使用的公开字段。"""
    if snapshot is None:
        return None
    from knowledge_agent.agent.planning.planner import validate_plan_summary

    return PlanSummaryResponse.model_validate(validate_plan_summary(snapshot))


def _answer_response(session_id: str, result: AnswerResult) -> AnswerResponse:
    """转换回答，不序列化任何 Agent 或模型状态。"""
    return AnswerResponse(
        session_id=session_id,
        question=result.question,
        answer=result.answer,
        status=result.status,
        sources=[_source_response(source) for source in result.sources],
        index_version=result.index_version,
        model_calls=result.model_calls,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        answer_diagnostics=public_answer_diagnostics(result.answer_diagnostics),
        trace=_trace_response(result.trace),
        routing=(
            RouteSummaryResponse.model_validate(result.routing, from_attributes=True)
            if result.routing is not None
            else None
        ),
        dialogue=(
            DialogueSummaryResponse.model_validate(result.dialogue, from_attributes=True)
            if result.dialogue is not None else None
        ),
        business_state=(
            BusinessStateResponse.model_validate(result.business_state, from_attributes=True)
            if result.business_state is not None else None
        ),
        assessment=_assessment_response(result.assessment_state),
        plan=_plan_response(result.plan),
    )


def _turn_response(turn: StoredTurn) -> TurnResponse:
    """原样暴露已存储的结果，同时省略其内部消息。"""
    result = turn.result
    return TurnResponse(
        turn_id=turn.turn_id,
        turn_no=turn.turn_no,
        question=turn.question,
        answer=result.answer,
        status=result.status,
        sources=[_source_response(source) for source in result.sources],
        index_version=result.index_version,
        model_calls=result.model_calls,
        tool_calls=result.tool_calls,
        stop_reason=result.stop_reason,
        answer_diagnostics=public_answer_diagnostics(result.answer_diagnostics),
        trace=_trace_response(result.trace),
        routing=(
            RouteSummaryResponse.model_validate(result.routing, from_attributes=True)
            if result.routing is not None
            else None
        ),
        dialogue=(
            DialogueSummaryResponse.model_validate(result.dialogue, from_attributes=True)
            if result.dialogue is not None else None
        ),
        business_state=(
            BusinessStateResponse.model_validate(result.business_state, from_attributes=True)
            if result.business_state is not None else None
        ),
        assessment=_assessment_response(result.assessment_state),
        plan=_plan_response(result.plan),
    )


def _document_response(document: DocumentSummary) -> DocumentResponse:
    response = DocumentResponse.model_validate(document, from_attributes=True)
    if response.error is None:
        return response
    return response.model_copy(update={"error": "The document could not be read"})


def _index_status_response(result: IndexStatus) -> IndexStatusResponse:
    changes = result.pending_changes
    return IndexStatusResponse(
        disk_status=result.disk_status,
        published_version=result.published_version,
        built_at=result.built_at,
        disk_document_count=result.disk_document_count,
        indexed_document_count=result.indexed_document_count,
        chunk_count=result.chunk_count,
        pending_changes=DocumentChangesResponse(
            added=changes.added,
            modified=changes.modified,
            removed=changes.removed,
            unreadable=changes.unreadable,
            total=changes.total,
        ),
        active_version=result.active_version,
        error=(
            None
            if result.error is None
            else {
                "not_built": "The knowledge index has not been built",
                "incomplete": "The published index is incomplete",
                "corrupt": "The published index metadata is invalid",
            }.get(result.disk_status, "The index status could not be determined")
        ),
    )


def _index_operation_response(result: IndexOperationResult) -> IndexOperationResponse:
    ingest = result.ingest_result
    return IndexOperationResponse(
        operation=result.operation,
        succeeded=result.succeeded,
        published=result.published,
        refreshed=result.refreshed,
        published_version=result.published_version,
        active_version=result.active_version,
        discovered_files=getattr(ingest, "discovered_files", None),
        read_files=getattr(ingest, "read_files", None),
        chunk_count=getattr(ingest, "chunk_count", None),
        changed_file_count=getattr(ingest, "changed_file_count", None),
        full_rebuild=getattr(ingest, "full_rebuild", None),
        reused_chunks=getattr(ingest, "reused_chunks", None),
        embedded_chunks=getattr(ingest, "embedded_chunks", None),
        removed_chunks=getattr(ingest, "removed_chunks", None),
        error_category=result.error_category,
    )
