from app.config import Settings
from app.db import ExperimentConfig
from app.providers.anthropic_provider import AnthropicChatProvider
from app.providers.base import ChatProvider, EmbeddingProvider
from app.providers.local_embedding_provider import SentenceTransformersEmbeddingProvider
from app.providers.ollama_provider import OllamaChatProvider, OllamaEmbeddingProvider
from app.providers.openai_provider import OpenAIChatProvider, OpenAIEmbeddingProvider

_EMBEDDING_MODEL_FIELD = {
    "openai": "openai_embedding_model",
    "ollama": "ollama_embedding_model",
    "local": "local_embedding_model",
}


def get_chat_provider(settings: Settings) -> ChatProvider:
    if settings.chat_provider == "anthropic":
        return AnthropicChatProvider(settings)
    if settings.chat_provider == "openai":
        return OpenAIChatProvider(settings)
    if settings.chat_provider == "ollama":
        return OllamaChatProvider(settings)
    raise ValueError(f"Unknown CHAT_PROVIDER: {settings.chat_provider!r}")


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(settings)
    if settings.embedding_provider == "local":
        return SentenceTransformersEmbeddingProvider(settings)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")


def get_embedding_provider_for_config(
    settings: Settings, config: ExperimentConfig
) -> EmbeddingProvider:
    if config.embedding_provider not in _EMBEDDING_MODEL_FIELD:
        raise ValueError(f"Unknown embedding_provider on config: {config.embedding_provider!r}")
    field = _EMBEDDING_MODEL_FIELD[config.embedding_provider]
    patched = settings.model_copy(
        update={"embedding_provider": config.embedding_provider, field: config.embedding_model}
    )
    return get_embedding_provider(patched)
