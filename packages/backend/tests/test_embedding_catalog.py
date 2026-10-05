from app.providers.embedding_catalog import get_embedding_dim


def test_get_embedding_dim_for_local_provider():
    assert get_embedding_dim("local", "BAAI/bge-base-en-v1.5") == 768


def test_get_embedding_dim_for_openai_provider():
    assert get_embedding_dim("openai", "text-embedding-3-large") == 3072


def test_get_embedding_dim_for_ollama_provider():
    assert get_embedding_dim("ollama", "mxbai-embed-large") == 1024


def test_get_embedding_dim_unknown_model_returns_none():
    assert get_embedding_dim("local", "not-a-real-model") is None


def test_get_embedding_dim_unknown_provider_returns_none():
    assert get_embedding_dim("not-a-real-provider", "text-embedding-3-large") is None
