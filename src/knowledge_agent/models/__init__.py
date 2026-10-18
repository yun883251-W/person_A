"""模型适配包；保留原 knowledge_agent.models 公共导入路径。"""
from importlib import import_module
from typing import Any

__all__ = [
    "ChatModelInvocationError",
    "ChatModelAuthenticationError",
    "ChatModelNotFoundError",
    "ChatModelTimeoutError",
    "ChatModelServiceError",
    "ChatModelRequestError",
    "ChatModelOutputError",
    "create_embedding_model",
    "create_chat_model",
    "initialize_chat_model",
    "invoke_chat_model",
    "bind_chat_json_schema",
    "bind_chat_tools",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module(".provider", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
