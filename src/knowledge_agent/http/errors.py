"""HTTP 业务错误映射；JSON 和 SSE 共用相同的公开错误结构。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..application import ApplicationBusyError, TurnPersistenceError
from ..models import ChatModelAuthenticationError, ChatModelInvocationError, ChatModelTimeoutError
from ..knowledge.manager import (
    DocumentAlreadyExistsError,
    DocumentReferenceMismatchError,
    DocumentTooLargeError,
    IncompleteIndexError,
    InvalidDocumentPathError,
    InvalidIndexManifestError,
    InvalidUploadEncodingError,
    InvalidUploadFilenameError,
    UploadTooLargeError,
)
from ..session_store import (
    DuplicateSessionError,
    DuplicateTurnError,
    SessionNotFoundError,
    SessionStoreError,
    SessionStoreLockedError,
)
from .runtime import ServiceBusyError
from .schemas import AnswerResponse, ErrorResponse
from .serialization import _answer_response

def _error_content(
    code: str,
    message: str,
    *,
    result: AnswerResponse | None = None,
    persisted: bool | None = None,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return ErrorResponse(
        code=code,
        message=message,
        result=result,
        persisted=persisted,
        details=details,
    ).model_dump(mode="json", exclude_none=True)


def _answer_service_error(response: AnswerResponse) -> JSONResponse:
    """Share business-error mapping between complete JSON and SSE responses."""
    if response.stop_reason == "index_not_built":
        status_code, code, message = (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "index_not_built",
            "The knowledge index has not been built; run `person-a build` and restart the API service",
        )
    elif response.stop_reason == "model_timeout":
        status_code, code, message = (
            status.HTTP_504_GATEWAY_TIMEOUT, "model_timeout", "The model response timed out",
        )
    elif response.stop_reason == "model_authentication_failed":
        status_code, code, message = (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "service_configuration_error",
            "The model service is not configured correctly",
        )
    else:
        status_code, code, message = (
            status.HTTP_503_SERVICE_UNAVAILABLE, "service_error", "The model service is unavailable",
        )
    return JSONResponse(
        status_code=status_code,
        content=_error_content(code, message, result=response),
    )


async def internal_error_handler(
    _request: Request, _exc: Exception
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_content("internal_error", "An internal service error occurred"),
    )


def register_error_handlers(application: FastAPI) -> None:
    """Attach exception mappings to an assembled HTTP application."""
    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "loc": list(issue.get("loc", ())),
                "msg": str(issue.get("msg", "Invalid value")),
                "type": str(issue.get("type", "value_error")),
            }
            for issue in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_content(
                "validation_error",
                "The request parameters are invalid",
                details=details,
            ),
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        if (
            exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            and isinstance(exc.detail, str)
            and exc.detail.startswith("Unknown skill:")
        ):
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_content("unknown_skill", exc.detail),
                headers=exc.headers,
            )
        known_errors = {
            status.HTTP_404_NOT_FOUND: ("session_not_found", "Session not found"),
            status.HTTP_409_CONFLICT: ("conflict", "The request conflicts with current state"),
            status.HTTP_422_UNPROCESSABLE_ENTITY: (
                "validation_error",
                "The request parameters are invalid",
            ),
        }
        code, message = known_errors.get(
            exc.status_code, ("http_error", "The request could not be completed")
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_content(code, message),
            headers=exc.headers,
        )

    @application.exception_handler(TurnPersistenceError)
    async def turn_persistence_error_handler(
        request: Request, exc: TurnPersistenceError
    ) -> JSONResponse:
        locked = isinstance(exc.__cause__, SessionStoreLockedError)
        return JSONResponse(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if locked
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content=_error_content(
                "persistence_unavailable" if locked else "persistence_failed",
                "The answer was generated, but the conversation turn was not saved",
                persisted=False,
                result=_answer_response(
                    str(request.path_params.get("session_id", "unknown")),
                    exc.result,
                ),
            ),
        )

    async def conflict_error_handler(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=_error_content(
                "conflict", "The session or conversation turn conflicts with current state"
            ),
        )

    for exception_type in (
        ApplicationBusyError,
        DuplicateSessionError,
        DuplicateTurnError,
    ):
        application.add_exception_handler(exception_type, conflict_error_handler)

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        _request: Request, _exc: SessionNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_content("session_not_found", "Session not found"),
        )

    async def unavailable_error_handler(
        _request: Request, _exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content(
                "service_unavailable", "The service is temporarily unavailable"
            ),
        )

    for exception_type in (ServiceBusyError, SessionStoreLockedError):
        application.add_exception_handler(exception_type, unavailable_error_handler)

    def knowledge_error(
        status_code: int, code: str, message: str
    ) -> Callable[[Request, Exception], Any]:
        async def handler(_request: Request, _exc: Exception) -> JSONResponse:
            return JSONResponse(
                status_code=status_code,
                content=_error_content(code, message),
            )

        return handler

    knowledge_error_mappings = (
        (FileNotFoundError, 404, "document_not_found", "Document not found"),
        (InvalidDocumentPathError, 422, "invalid_document_path", "The document path is invalid"),
        (DocumentReferenceMismatchError, 409, "document_changed", "The document no longer matches the referenced content"),
        (DocumentAlreadyExistsError, 409, "document_exists", "A document with that name already exists"),
        (InvalidUploadFilenameError, 422, "invalid_filename", "The upload filename is invalid"),
        (InvalidUploadEncodingError, 422, "invalid_encoding", "The document must use UTF-8 encoding"),
        (UploadTooLargeError, 413, "upload_too_large", "The uploaded document is too large"),
        (DocumentTooLargeError, 413, "document_too_large", "The document is too large to preview"),
        (IncompleteIndexError, 500, "index_incomplete", "The published index is incomplete"),
        (InvalidIndexManifestError, 500, "index_corrupt", "The published index metadata is invalid"),
        (PermissionError, 500, "knowledge_storage_error", "Knowledge-base storage is unavailable"),
        (NotADirectoryError, 500, "knowledge_storage_error", "Knowledge-base storage is unavailable"),
    )
    for exception_type, status_code, code, message in knowledge_error_mappings:
        application.add_exception_handler(
            exception_type, knowledge_error(status_code, code, message)
        )

    @application.exception_handler(ChatModelTimeoutError)
    async def model_timeout_handler(
        _request: Request, _exc: ChatModelTimeoutError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=_error_content("model_timeout", "The model response timed out"),
        )

    @application.exception_handler(ChatModelAuthenticationError)
    async def model_authentication_error_handler(
        _request: Request, _exc: ChatModelAuthenticationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content(
                "service_configuration_error",
                "The model service is not configured correctly",
            ),
        )

    @application.exception_handler(ChatModelInvocationError)
    async def model_invocation_error_handler(
        _request: Request, _exc: ChatModelInvocationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_content("service_error", "The model service is unavailable"),
        )

    @application.exception_handler(SessionStoreError)
    async def storage_error_handler(
        _request: Request, _exc: SessionStoreError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_content("storage_error", "Session storage failed"),
        )

    application.add_exception_handler(Exception, internal_error_handler)
