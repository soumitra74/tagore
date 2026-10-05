import pytest

from app.config import Settings
from app.db import ExperimentConfig
from app.providers.anthropic_provider import AnthropicChatProvider
from app.providers.factory import (
    get_chat_provider,
    get_embedding_provider,
    get_embedding_provider_for_config,
)
from app.providers.local_embedding_provider import SentenceTransformersEmbeddingProvider
from app.providers.ollama_provider import OllamaChatProvider, OllamaEmbeddingProvider
from app.providers.openai_provider import OpenAIChatProvider, OpenAIEmbeddingProvider


def test_get_chat_provider_anthropic():
    settings = Settings(chat_provider="anthropic", anthropic_api_key="test-key")
    assert isinstance(get_chat_provider(settings), AnthropicChatProvider)


def test_get_chat_provider_openai():
    settings = Settings(chat_provider="openai", openai_api_key="test-key")
    assert isinstance(get_chat_provider(settings), OpenAIChatProvider)


def test_get_chat_provider_ollama():
    settings = Settings(chat_provider="ollama")
    assert isinstance(get_chat_provider(settings), OllamaChatProvider)


def test_get_chat_provider_unknown_raises():
    settings = Settings(chat_provider="bogus")
    with pytest.raises(ValueError):
        get_chat_provider(settings)


def test_get_embedding_provider_openai():
    settings = Settings(embedding_provider="openai", openai_api_key="test-key")
    assert isinstance(get_embedding_provider(settings), OpenAIEmbeddingProvider)


def test_get_embedding_provider_ollama():
    settings = Settings(embedding_provider="ollama")
    assert isinstance(get_embedding_provider(settings), OllamaEmbeddingProvider)


def test_get_embedding_provider_unknown_raises():
    settings = Settings(embedding_provider="bogus")
    with pytest.raises(ValueError):
        get_embedding_provider(settings)


def test_get_embedding_provider_local():
    settings = Settings(embedding_provider="local")
    provider = get_embedding_provider(settings)
    assert isinstance(provider, SentenceTransformersEmbeddingProvider)


def test_get_embedding_provider_local_unknown_model_raises():
    settings = Settings(embedding_provider="local", local_embedding_model="bogus/model")
    with pytest.raises(ValueError):
        get_embedding_provider(settings)


def test_get_embedding_provider_for_config_local():
    settings = Settings(embedding_provider="openai", openai_api_key="test-key")
    config = ExperimentConfig(
        id=1,
        chunking_strategy="sliding_window",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_provider="local",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
    )
    provider = get_embedding_provider_for_config(settings, config)
    assert isinstance(provider, SentenceTransformersEmbeddingProvider)


def test_get_embedding_provider_for_config_uses_configs_model_not_settings_default():
    settings = Settings(embedding_provider="local", local_embedding_model="BAAI/bge-base-en-v1.5")
    config = ExperimentConfig(
        id=2,
        chunking_strategy="sliding_window",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_provider="local",
        embedding_model="intfloat/e5-base-v2",
        embedding_dim=768,
    )
    provider = get_embedding_provider_for_config(settings, config)
    assert provider._model_name == "intfloat/e5-base-v2"


class _FakeSentenceTransformer:
    def __init__(self, model_name, trust_remote_code=False):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.max_seq_length = 512  # sentence-transformers' typical auto-detected default


def test_get_model_raises_max_seq_length_for_bge_m3(monkeypatch):
    monkeypatch.setattr(
        "app.providers.local_embedding_provider.SentenceTransformer", _FakeSentenceTransformer
    )
    settings = Settings(embedding_provider="local", local_embedding_model="BAAI/bge-m3")
    provider = get_embedding_provider(settings)

    model = provider._get_model()

    assert model.max_seq_length == 8192


def test_get_model_leaves_max_seq_length_untouched_when_not_overridden(monkeypatch):
    monkeypatch.setattr(
        "app.providers.local_embedding_provider.SentenceTransformer", _FakeSentenceTransformer
    )
    settings = Settings(embedding_provider="local", local_embedding_model="BAAI/bge-base-en-v1.5")
    provider = get_embedding_provider(settings)

    model = provider._get_model()

    assert model.max_seq_length == 512


def test_get_embedding_provider_for_config_unknown_provider_raises():
    settings = Settings(embedding_provider="local")
    config = ExperimentConfig(
        id=3,
        chunking_strategy="sliding_window",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_provider="bogus",
        embedding_model="whatever",
        embedding_dim=768,
    )
    with pytest.raises(ValueError):
        get_embedding_provider_for_config(settings, config)
