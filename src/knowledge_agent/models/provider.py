"""模型初始化。

分别创建对话模型和 Embedding 模型，支持各自的服务地址与凭据。
建立文档索引和查询问题必须采用兼容的同一套 Embedding 配置。
Embedding 模型、维度或处理方式改变时，应重新建立索引。

提供方、型号和参数从现有配置读取，对话与 Embedding 分开选择。
本模块由原 models.py 迁移，调用和错误处理逻辑保持不变。
原 knowledge_agent.models 的公共 API 由包入口延迟导出。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Mapping, cast
from urllib.parse import urlsplit

from knowledge_agent.models.diagnostics import record_chat_failure, record_embedding_timing

if TYPE_CHECKING:
    from knowledge_agent.config import ChatSettings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage


class ChatModelInvocationError(RuntimeError):
    """对话模型请求无法完成时抛出的基类错误。"""


class ChatModelAuthenticationError(ChatModelInvocationError):
    """提供方拒绝了已配置的凭据。"""


class ChatModelNotFoundError(ChatModelInvocationError):
    """配置的模型或端点不存在。"""


class ChatModelTimeoutError(ChatModelInvocationError):
    """提供方请求超过配置的超时时间。"""


class ChatModelServiceError(ChatModelInvocationError):
    """暂时性的提供方或网络故障在重试后仍未恢复。"""


class ChatModelRequestError(ChatModelInvocationError):
    """请求无效，或使用了模型不具备的能力。"""


class ChatModelOutputError(ChatModelInvocationError):
    """提供方已响应，但响应或工具参数无效。"""


def create_embedding_model(
    *, provider: str, model: str, base_url: str, request_timeout: float = 60.0
) -> Any:
    """创建已配置的 Embedding 适配器，但不联系服务。"""
    if provider.strip().lower() != "ollama":
        raise ValueError(f"不支持的 Embedding provider: {provider}")
    if not model.strip():
        raise ValueError("EMBEDDING_MODEL 不能为空")
    if not base_url.strip():
        raise ValueError("EMBEDDING_BASE_URL 不能为空")

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Ollama Embedding 依赖，请安装项目依赖 langchain-ollama"
        ) from exc

    class NonTruncatingOllamaEmbeddings(OllamaEmbeddings):
        """绝不静默截断文档输入的 Ollama 适配器。"""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            response = self._client.embed(
                self.model,
                texts,
                truncate=False,
                options=self._default_params,
                keep_alive=self.keep_alive,
            )
            record_embedding_timing(response)
            return response["embeddings"]

    return NonTruncatingOllamaEmbeddings(
        model=model.strip(),
        base_url=base_url.rstrip("/"),
        client_kwargs={"timeout": request_timeout},
    )


# 第一阶段：补齐对话模型初始化。保留上方已有的 Embedding 接口。


def create_chat_model(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str | None = None,
    request_timeout: float = 60.0,
    max_retries: int = 2,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> BaseChatModel:
    """为 Agent 消息输入和结构化输出创建对话适配器。

    该工厂只负责适配显式传入的提供方设置。
    提示词构造、检索、对话编排、工具执行以及完整业务回答的生成
    都属于各自所在的层。
    """
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("CHAT_PROVIDER 不能为空")
    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"deepseek", "ollama"}:
        raise ValueError(f"不支持的对话模型 provider: {provider.strip()}")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("CHAT_MODEL 不能为空")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("CHAT_BASE_URL 不能为空")
    normalized_base_url = base_url.strip()
    try:
        parsed_url = urlsplit(normalized_base_url)
        parsed_url.port
    except ValueError as exc:
        raise ValueError(f"CHAT_BASE_URL 格式无效: {base_url}") from exc
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(f"CHAT_BASE_URL 必须是有效的 HTTP(S) 服务地址: {base_url}")

    if isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float)):
        raise ValueError("CHAT_REQUEST_TIMEOUT 必须是大于 0 的有限数值")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("CHAT_REQUEST_TIMEOUT 必须是大于 0 的有限数值")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError("CHAT_MAX_RETRIES 必须是 0 到 10 之间的整数")
    if not 0 <= max_retries <= 10:
        raise ValueError("CHAT_MAX_RETRIES 必须是 0 到 10 之间的整数")

    if normalized_provider == "deepseek":
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("DeepSeek provider 需要 CHAT_API_KEY")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("DeepSeek CHAT_TEMPERATURE 必须在 0 到 2 之间")
    elif temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("Ollama CHAT_TEMPERATURE 必须是非负有限数值")

    if max_output_tokens is not None:
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
            raise ValueError("CHAT_MAX_OUTPUT_TOKENS 必须是整数")
        if normalized_provider == "deepseek" and max_output_tokens <= 0:
            raise ValueError("DeepSeek CHAT_MAX_OUTPUT_TOKENS 必须大于 0")
        if (
            normalized_provider == "ollama"
            and max_output_tokens not in {-2, -1}
            and max_output_tokens <= 0
        ):
            raise ValueError("Ollama CHAT_MAX_OUTPUT_TOKENS 必须大于 0，或为 -1/-2")

    if model_kwargs is not None and not isinstance(model_kwargs, Mapping):
        raise ValueError("model_kwargs 必须是映射")

    if normalized_provider == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as exc:
            raise RuntimeError(
                "缺少 DeepSeek 对话模型依赖，请安装项目依赖 langchain-deepseek"
            ) from exc

        deepseek_options: dict[str, Any] = {
            "model": model.strip(),
            "api_key": api_key.strip(),
            "base_url": normalized_base_url.rstrip("/"),
            "timeout": float(request_timeout),
            # 重试统一集中在 invoke_chat_model，使所有提供方共享
            # 同一套有限策略，避免嵌套重试成倍放大。
            "max_retries": 0,
        }
        if temperature is not None:
            deepseek_options["temperature"] = temperature
        if max_output_tokens is not None:
            deepseek_options["max_tokens"] = max_output_tokens
        if model_kwargs:
            deepseek_options["model_kwargs"] = dict(model_kwargs)
        chat_model = ChatDeepSeek(**deepseek_options)
        object.__setattr__(chat_model, "_knowledge_agent_max_retries", max_retries)
        return chat_model

    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Ollama 对话模型依赖，请安装项目依赖 langchain-ollama"
        ) from exc

    ollama_options = dict(model_kwargs or {})
    ollama_options.update(
        model=model.strip(),
        base_url=normalized_base_url.rstrip("/"),
        client_kwargs={"timeout": float(request_timeout)},
    )
    if temperature is not None:
        ollama_options["temperature"] = temperature
    if max_output_tokens is not None:
        ollama_options["num_predict"] = max_output_tokens
    chat_model = ChatOllama(**ollama_options)
    object.__setattr__(chat_model, "_knowledge_agent_max_retries", max_retries)
    return chat_model

def initialize_chat_model() -> BaseChatModel:
    """返回当前对话配置对应的客户端。

    只要对话设置未变就复用同一客户端。配置变化时会创建并保留
    替代客户端；Embedding 客户端和索引刻意不在此生命周期内。
    """
    from knowledge_agent.config import get_chat_settings

    from .concurrency import current_gateway
    gateway = current_gateway()
    if gateway is not None:
        return gateway.model_for(get_chat_settings(), _configured_chat_model.__wrapped__)
    return _configured_chat_model(get_chat_settings())


@lru_cache(maxsize=1)
def _configured_chat_model(settings: ChatSettings) -> BaseChatModel:
    """为每个生效的对话配置创建一个无状态、可复用的客户端。"""
    return create_chat_model(
        provider=settings.provider,
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        request_timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
    )

def invoke_chat_model(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    max_retries: int | None = None,
) -> AIMessage:
    """同步返回一条完整消息，包括工具调用元数据。

    HTTP 工作通道通过协程网关等待 ``ainvoke``；CLI 继续使用 ``invoke``。
    调用方只拿到提供方已完成的消息，工具执行仍留在 Agent 层。
    """
    retries = (
        getattr(model, "_knowledge_agent_max_retries", 0)
        if max_retries is None
        else max_retries
    )
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    for attempt in range(retries + 1):
        try:
            from .concurrency import invoke_with_gateway
            return cast("AIMessage", invoke_with_gateway(model, messages))
        except Exception as exc:
            error, recoverable = _normalize_chat_model_error(exc)
            record_chat_failure(
                exc, status_code=_exception_status_code(exc),
                error_category=type(error).__name__, retryable=recoverable,
                attempt=attempt + 1, model=model,
            )
            if not recoverable or attempt == retries:
                raise error from exc

    raise AssertionError("unreachable")


def _normalize_chat_model_error(
    exc: Exception,
) -> tuple[ChatModelInvocationError, bool]:
    """在不导入提供方 SDK 的情况下转换提供方特有的失败。"""
    status = _exception_status_code(exc)
    name = type(exc).__name__.lower()

    if "outputparser" in name:
        return ChatModelOutputError("The model returned an invalid structured response."), False
    if status in {401, 403} or "authentication" in name or "permissiondenied" in name:
        return ChatModelAuthenticationError("Chat model authentication failed."), False
    if status == 404 or "notfound" in name:
        return ChatModelNotFoundError("The configured chat model was not found."), False
    if status in {408, 504} or isinstance(exc, TimeoutError) or "timeout" in name:
        return ChatModelTimeoutError("The chat model request timed out."), True
    if (
        status in {409, 429}
        or (status is not None and status >= 500)
        or isinstance(exc, ConnectionError)
        or any(token in name for token in ("connection", "ratelimit", "serviceunavailable"))
    ):
        return ChatModelServiceError(
            "The chat model service is temporarily unavailable."
        ), True
    if status is not None and 400 <= status < 500:
        return ChatModelRequestError("The chat model rejected the request."), False
    if isinstance(exc, (TypeError, ValueError, NotImplementedError)):
        return ChatModelRequestError(
            "The chat model request is invalid or unsupported."
        ), False
    return ChatModelInvocationError("The chat model request failed."), False


def _exception_status_code(exc: Exception) -> int | None:
    """尽力从 OpenAI/httpx/Ollama 风格的异常中提取状态码。"""
    for candidate in (exc, getattr(exc, "response", None)):
        for attribute in ("status_code", "status"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def bind_chat_json_schema(model: BaseChatModel, schema: Mapping[str, Any]) -> Any:
    """在不改变客户端的前提下约束一次隔离的 Ollama JSON 响应。

    使用原生 ``format`` schema，而不是合成工具或宽松的 JSON 解析器。
    调用方仍负责严格解析、语义校验和预算控制。返回绑定后的 runnable
    会保留原始 AIMessage，并让共享模型上的普通回答和工具调用不受影响。
    其他提供方保持现有的基于提示词的契约；不要发送 Ollama 专属参数。
    该函数只准备请求，绝不联系服务。
    """
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("The response schema must describe a JSON object")
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return model
    if not isinstance(model, ChatOllama):
        return model
    try:
        return model.bind(format=deepcopy(dict(schema)))
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ChatModelRequestError(
            "The configured Ollama adapter does not support structured output."
        ) from exc


def bind_chat_tools(model: BaseChatModel, tools: Sequence[Any]) -> Any:
    """绑定由 Agent 提供的工具 schema，但不执行工具。

    工具调用执行、结果消息和迭代上限仍由 Agent 层负责。
    """
    try:
        return model.bind_tools(tools)
    except NotImplementedError as exc:
        raise ChatModelRequestError(
            "The configured chat model does not support tool calling."
        ) from exc

CHAT_STREAMING_SUPPORTED: Final = False
