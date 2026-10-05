from openai import AsyncOpenAI

from app.config import Settings
from app.providers.base import LLMResponse, build_prompt


class OpenAIChatProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=300)
        self._model = settings.openai_chat_model

    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        prompt = build_prompt(question, context_chunks)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class OpenAIEmbeddingProvider:
    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        [embedding] = await self.embed_documents([text])
        return embedding
