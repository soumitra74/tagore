import pytest

from config import Settings
from llm_client import (
    AnthropicCompletionProvider,
    OllamaCompletionProvider,
    OpenAICompletionProvider,
    get_completion_provider,
)


def test_get_completion_provider_anthropic():
    settings = Settings(_env_file=None, eval_judge_provider="anthropic", anthropic_api_key="test-key")
    assert isinstance(get_completion_provider(settings), AnthropicCompletionProvider)


def test_get_completion_provider_openai():
    settings = Settings(_env_file=None, eval_judge_provider="openai", openai_api_key="test-key")
    assert isinstance(get_completion_provider(settings), OpenAICompletionProvider)


def test_get_completion_provider_ollama():
    settings = Settings(_env_file=None, eval_judge_provider="ollama")
    assert isinstance(get_completion_provider(settings), OllamaCompletionProvider)


def test_get_completion_provider_unknown_raises():
    settings = Settings(_env_file=None, eval_judge_provider="bogus")
    with pytest.raises(ValueError):
        get_completion_provider(settings)
