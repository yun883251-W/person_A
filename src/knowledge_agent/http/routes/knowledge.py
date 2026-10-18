"""知识库文档、上传和索引管理路由。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Query, status
from fastapi.responses import JSONResponse

from ...knowledge.manager import UploadTooLargeError
from ..errors import _error_content
from ..runtime import ServiceRuntime
from ..schemas import (
    MAX_PREVIEW_LINES,
    MAX_UPLOAD_BYTES,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentResponse,
    IndexOperationResponse,
    IndexStatusResponse,
    PageLimit,
    PageOffset,
)
from ..serialization import _document_response, _index_operation_response, _index_status_response


def register_routes(application: FastAPI) -> None:
    """在共享服务运行时上注册文档与索引操作路由。"""
    @application.get(
        "/knowledge/documents",
        response_model=DocumentListResponse,
        summary="分页列出知识库文档",
    )
    async def list_knowledge_documents(
        limit: PageLimit = 50,
        offset: PageOffset = 0,
    ) -> DocumentListResponse:
        runtime: ServiceRuntime = application.state.runtime
        documents = await runtime.run_knowledge(
            lambda service: service.list_documents(offset=offset, limit=limit)
        )
        return DocumentListResponse(
            items=[_document_response(item) for item in documents],
            limit=limit,
            offset=offset,
        )

    @application.get(
        "/knowledge/documents/content",
        response_model=DocumentPreviewResponse,
        summary="预览受控知识库原文",
    )
    async def read_knowledge_document(
        path: Annotated[str, Query(min_length=1)],
        start_line: Annotated[int, Query(ge=1)] = 1,
        limit: Annotated[int, Query(ge=1, le=MAX_PREVIEW_LINES)] = 200,
        content_hash: Annotated[str | None, Query(min_length=1)] = None,
    ) -> DocumentPreviewResponse:
        runtime: ServiceRuntime = application.state.runtime
        preview = await runtime.run_knowledge(
            lambda service: service.read_document(
                path,
                start_line=start_line,
                limit=limit,
                content_hash=content_hash,
            )
        )
        return DocumentPreviewResponse.model_validate(preview, from_attributes=True)

    @application.get(
        "/knowledge/status",
        response_model=IndexStatusResponse,
        summary="读取知识库索引状态",
    )
    async def knowledge_status() -> IndexStatusResponse:
        runtime: ServiceRuntime = application.state.runtime
        return _index_status_response(await runtime.get_knowledge_status())

    @application.post(
        "/knowledge/documents",
        response_model=DocumentResponse,
        status_code=status.HTTP_201_CREATED,
        summary="上传新的 Markdown 文档",
    )
    async def upload_knowledge_document(
        filename: Annotated[str, Query(min_length=1, max_length=255)],
        content: Annotated[bytes, Body(media_type="text/markdown")],
    ) -> DocumentResponse:
        if len(content) > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError("upload exceeds the HTTP body limit")
        runtime: ServiceRuntime = application.state.runtime
        document = await runtime.run_knowledge(
            lambda service: service.upload_document(filename, content)
        )
        return _document_response(document)

    async def run_index_operation(operation: Literal["build", "update"]) -> Any:
        runtime: ServiceRuntime = application.state.runtime
        result = await runtime.run_knowledge(
            lambda service: (
                service.build_index() if operation == "build" else service.update_index()
            )
        )
        response = _index_operation_response(result)
        if result.succeeded:
            return response
        refresh_failed = result.error_category == "refresh"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                **_error_content(
                    "index_refresh_failed" if refresh_failed else "index_ingest_failed",
                    (
                        "The index was published, but the running service could not reload it"
                        if refresh_failed
                        else "The index operation failed"
                    ),
                ),
                "operation": response.model_dump(mode="json", exclude_none=True),
            },
        )

    @application.post(
        "/knowledge/index/build",
        response_model=IndexOperationResponse,
        summary="完整重建知识库索引",
    )
    async def build_knowledge_index() -> Any:
        return await run_index_operation("build")

    @application.post(
        "/knowledge/index/update",
        response_model=IndexOperationResponse,
        summary="增量更新知识库索引",
    )
    async def update_knowledge_index() -> Any:
        return await run_index_operation("update")
