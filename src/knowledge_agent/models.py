"""模型初始化。

分别创建对话模型和 Embedding 模型，支持各自的服务地址与凭据。
建立文档索引和查询问题必须采用兼容的同一套 Embedding 配置。
Embedding 模型、维度或处理方式改变时，应重新建立索引。
"""

from __future__ import annotations

from typing import Any


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
