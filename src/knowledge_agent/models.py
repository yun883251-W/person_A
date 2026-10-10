"""模型初始化。

分别创建对话模型和 Embedding 模型，支持各自的服务地址与凭据。
建立文档索引和查询问题必须采用兼容的同一套 Embedding 配置。
Embedding 模型、维度或处理方式改变时，应重新建立索引。

当前已有 create_embedding_model；下方 TODO 用于补齐对话模型部分。
对话模型的提供方与具体型号尚未确定，不能从 Embedding 配置推断。
先完成 TODO 01—10 的开发，再集中执行 TODO 11—14 的验证。
配置检查属于待实现的运行时功能；本阶段不执行联网探测、模型调用或测试。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Mapping, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .config import ChatSettings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage


class ChatModelInvocationError(RuntimeError):
    """Base error raised when a chat-model request cannot be completed."""


class ChatModelAuthenticationError(ChatModelInvocationError):
    """The provider rejected the configured credentials."""


class ChatModelNotFoundError(ChatModelInvocationError):
    """The configured model or endpoint does not exist."""


class ChatModelTimeoutError(ChatModelInvocationError):
    """The provider request exceeded its configured timeout."""


class ChatModelServiceError(ChatModelInvocationError):
    """A transient provider or network failure exhausted its retries."""


class ChatModelRequestError(ChatModelInvocationError):
    """The request is invalid or uses a capability the model lacks."""


def create_embedding_model(
    *, provider: str, model: str, base_url: str, request_timeout: float = 60.0
) -> Any:
    """Create the configured embedding adapter without contacting the service."""
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
        """Ollama adapter that never silently truncates document input."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            response = self._client.embed(
                self.model,
                texts,
                truncate=False,
                options=self._default_params,
                keep_alive=self.keep_alive,
            )
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
    """Create a chat adapter for agent message input and structured output.

    The factory is limited to adapting the explicitly supplied provider settings.
    Prompt construction, retrieval, conversation orchestration, tool execution, and
    generation of a complete business answer belong to their respective layers.
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
            # Retrying is centralized in invoke_chat_model so all providers have
            # the same finite policy and nested retries cannot multiply.
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
    """Return the client for the current chat configuration.

    A client is reused while all chat settings remain unchanged.  A changed
    configuration creates and retains a replacement client; embedding clients
    and indexes are deliberately outside this lifecycle.
    """
    from .config import get_chat_settings

    return _configured_chat_model(get_chat_settings())


@lru_cache(maxsize=1)
def _configured_chat_model(settings: ChatSettings) -> BaseChatModel:
    """Create one stateless, reusable client per active chat configuration."""
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
    """Synchronously return one complete message, including tool-call metadata.

    This boundary deliberately uses ``invoke`` rather than consuming ``stream``:
    callers only receive the provider's completed message, and tool execution stays
    in the agent layer.
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
            return cast("AIMessage", model.invoke(messages))
        except Exception as exc:
            error, recoverable = _normalize_chat_model_error(exc)
            if not recoverable or attempt == retries:
                raise error from exc

    raise AssertionError("unreachable")


def _normalize_chat_model_error(
    exc: Exception,
) -> tuple[ChatModelInvocationError, bool]:
    """Translate provider-specific failures without importing provider SDKs."""
    status = _exception_status_code(exc)
    name = type(exc).__name__.lower()

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
    """Best-effort extraction for OpenAI/httpx/Ollama-style exceptions."""
    for candidate in (exc, getattr(exc, "response", None)):
        for attribute in ("status_code", "status"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None

def bind_chat_tools(model: BaseChatModel, tools: Sequence[Any]) -> Any:
    """Bind agent-supplied tool schemas without executing the tools.

    Tool-call execution, result messages, and iteration limits remain the
    responsibility of the agent layer.
    """
    try:
        return model.bind_tools(tools)
    except NotImplementedError as exc:
        raise ChatModelRequestError(
            "The configured chat model does not support tool calling."
        ) from exc

CHAT_STREAMING_SUPPORTED: Final = False


# 第二阶段：开发全部完成后统一验证。以下 TODO 暂不执行。

# TODO 11：集中验证配置与工厂初始化。
# 在独立的模型测试文件中使用假客户端，覆盖配置读取优先级及参数映射。
# 覆盖空提供方、空模型名、无效地址、非法超时、缺少依赖与不支持的提供方。
# 确认聊天配置不会读取 Embedding 配置，且本地默认接口不强制 API Key。
# 确认导入及构造不会主动发送模型请求，也不会安装依赖或下载模型。

# TODO 12：集中验证消息、工具调用与异常处理。
# 使用可控响应覆盖普通文本、空正文但存在 tool_calls、工具结果关联及多轮消息。
# 确认结构化工具参数与调用 ID 保留完整，模型层不会自行执行工具。
# 模拟超时、认证错误、模型缺失和临时故障，检查重试次数及错误分类。
# 如实现流式能力，再覆盖工具参数分片、中途失败和避免重复输出的行为。

# TODO 13：集中执行真实对话模型联调。
# 确定并配置实际对话提供方及模型后，再检查服务连接和一次基础中文问答。
# 绑定一个返回固定内容的只读测试工具，验证模型能够提出结构化工具调用，
# 将对应工具结果交回模型后，能够继续生成回答；不能只验证普通聊天成功。
# 服务声明的能力与实际行为不一致时记录问题，不将未完成的能力标为可用。
# 当前只验证模型接入；知识库答案质量和来源引用留给后续 Agent 的集中联调。

# TODO 14：集中执行 Embedding 接口回归并记录验证结论。
# 确认新增聊天模型能力后，已有入库和检索仍可使用 create_embedding_model。
# 覆盖文档向量化、查询向量化、超长输入处理以及模型配置相互隔离。
# 确认仅修改聊天配置不会改变 Embedding 模型身份、向量维度或索引数据。
# 汇总已通过、未通过和未执行项目；未选择实际对话模型时保留相应待验证项。
