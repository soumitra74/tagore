import pytest

from app.db import ExperimentConfig
from app.providers.base import LLMResponse
from app.services.chat import NO_CONTEXT_ANSWER, answer_question


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeChatProvider:
    def __init__(self):
        self.received_context: list[str] | None = None

    async def generate(self, question: str, context_chunks: list[str]) -> LLMResponse:
        self.received_context = context_chunks
        return LLMResponse(text=f"answer to: {question}", input_tokens=100, output_tokens=20)


class FakeDatabaseWithResults:
    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [
            {"id": "11111111-1111-1111-1111-111111111111", "chunk_text": "chunk one",
             "context_text": None, "page_number": 3, "filename": "doc.pdf"},
            {"id": "22222222-2222-2222-2222-222222222222", "chunk_text": "chunk two",
             "context_text": None, "page_number": 4, "filename": "doc.pdf"},
        ]


class FakeDatabaseWithSentenceWindowResults:
    """context_text set, as a real sentence_window row would have -- chat.py
    should show the generator the padded context, not the bare sentence."""

    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [
            {"id": "11111111-1111-1111-1111-111111111111", "chunk_text": "The anchor sentence.",
             "context_text": "Prior sentence. The anchor sentence. Next sentence.",
             "page_number": 3, "filename": "doc.pdf"},
        ]


class FakeDatabaseWithNoResults:
    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return []


class FakeDatabaseWithDuplicates:
    """Two rows share near-identical text (mirrors the real corpus, where
    PFR1/PFR2 share boilerplate methodology paragraphs); one row is distinct.
    """

    def __init__(self):
        self.requested_limit: int | None = None

    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        self.requested_limit = top_k
        return [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "chunk_text": "The data quality analysis was performed in two parts.",
                "context_text": None,
                "page_number": 16,
                "filename": "PFR1.pdf",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "chunk_text": "The data quality analysis was performed in two parts .",
                "context_text": None,
                "page_number": 16,
                "filename": "PFR2.pdf",
            },
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "chunk_text": "An entirely unrelated paragraph about rotor blades.",
                "context_text": None,
                "page_number": 3,
                "filename": "PFR1.pdf",
            },
        ]


TRULY_DISTINCT_PARAGRAPHS = [
    "Rotor blade airloads were measured using pressure transducers.",
    "The flight test program spanned multiple altitude and airspeed conditions.",
    "Vibration data was collected from accelerometers mounted on the hub.",
    "Ground resonance testing preceded the first untethered flight.",
    "Data acquisition rates varied by channel type and criticality.",
    "Post-flight calibration checks validated sensor drift bounds.",
]


class FakeDatabaseWithManyResults:
    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [
            {"id": f"{i:08d}-0000-0000-0000-000000000000", "chunk_text": text,
             "context_text": None, "page_number": i, "filename": "doc.pdf"}
            for i, text in enumerate(TRULY_DISTINCT_PARAGRAPHS[:top_k])
        ]


TEST_CONFIG = ExperimentConfig(
    id=1,
    chunking_strategy="sliding_window",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_provider="local",
    embedding_model="BAAI/bge-base-en-v1.5",
    embedding_dim=768,
)


@pytest.mark.asyncio
async def test_answer_question_returns_answer_with_sources():
    chat_provider = FakeChatProvider()
    result = await answer_question(
        "what happened?",
        FakeDatabaseWithResults(),
        FakeEmbeddingProvider(),
        chat_provider,
        top_k=5,
        config=TEST_CONFIG,
    )

    assert result.answer == "answer to: what happened?"
    # The generator is given labeled context so it can cite sources by number...
    assert chat_provider.received_context == [
        "[Source 1 -- doc.pdf, page 3]\nchunk one",
        "[Source 2 -- doc.pdf, page 4]\nchunk two",
    ]
    assert len(result.sources) == 2
    assert result.sources[0].document == "doc.pdf"
    assert result.sources[0].page == 3
    assert result.sources[0].snippet == "chunk one"
    assert result.sources[0].chunk_id == "11111111-1111-1111-1111-111111111111"
    assert result.sources[1].chunk_id == "22222222-2222-2222-2222-222222222222"
    assert result.generation_input_tokens == 100
    assert result.generation_output_tokens == 20
    # ...but context_chunks (used for grounding judges) stays unlabeled.
    assert result.context_chunks == ["chunk one", "chunk two"]


@pytest.mark.asyncio
async def test_answer_question_prefers_context_text_over_chunk_text():
    # sentence_window rows carry context_text (the sentence padded with
    # neighbors) -- the generator and grounding judges should see that, not
    # the bare single sentence that was actually embedded.
    chat_provider = FakeChatProvider()
    result = await answer_question(
        "what happened?",
        FakeDatabaseWithSentenceWindowResults(),
        FakeEmbeddingProvider(),
        chat_provider,
        top_k=5,
        config=TEST_CONFIG,
    )

    assert chat_provider.received_context == [
        "[Source 1 -- doc.pdf, page 3]\nPrior sentence. The anchor sentence. Next sentence.",
    ]
    assert result.context_chunks == ["Prior sentence. The anchor sentence. Next sentence."]
    # The citation snippet still previews the precise embedded sentence.
    assert result.sources[0].snippet == "The anchor sentence."


@pytest.mark.asyncio
async def test_answer_question_with_no_matches_returns_fallback():
    result = await answer_question(
        "what happened?",
        FakeDatabaseWithNoResults(),
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=5,
        config=TEST_CONFIG,
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []


@pytest.mark.asyncio
async def test_answer_question_dedupes_near_duplicate_chunks():
    db = FakeDatabaseWithDuplicates()
    result = await answer_question(
        "what was the data quality analysis?",
        db,
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=5,
        config=TEST_CONFIG,
    )

    # The near-duplicate PFR2 row is dropped; the PFR1 duplicate and the
    # distinct row survive.
    assert len(result.sources) == 2
    documents = {s.document for s in result.sources}
    assert documents == {"PFR1.pdf"}
    # Retrieval over-fetches beyond top_k so there's enough headroom to
    # dedupe down to top_k distinct chunks.
    assert db.requested_limit > 5


@pytest.mark.asyncio
async def test_answer_question_respects_requested_top_k():
    result = await answer_question(
        "what happened?",
        FakeDatabaseWithManyResults(),
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=2,
        config=TEST_CONFIG,
    )

    assert len(result.sources) == 2


class FakeEntity:
    def __init__(self, entity_type, normalized_text):
        self.entity_type = entity_type
        self.normalized_text = normalized_text


class FakeNER:
    def __init__(self, entities):
        self._entities = entities

    async def extract(self, text):
        return self._entities


class FakeDatabaseWithGraphMatch:
    """Vector search alone misses a relevant chunk (ranked outside fetch_k);
    the graph path recovers it via an entity mention."""

    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [
            {"id": "11111111-1111-1111-1111-111111111111", "chunk_text": "unrelated chunk",
             "context_text": None, "page_number": 3, "filename": "PFR1.pdf", "distance": 0.1},
        ]

    async def find_entities(self, pairs):
        assert pairs == [("DATA_COUNTER_ID", "9017")]
        return ["entity-1"]

    async def get_chunks_mentioning_entities(self, config_id, dim, entity_ids, embedding, limit):
        assert entity_ids == ["entity-1"]
        return [
            {"id": "22222222-2222-2222-2222-222222222222", "chunk_text": "Counter 9017 data",
             "context_text": None, "page_number": 18, "filename": "PFR1.pdf", "distance": 0.05},
        ]


class FakeDatabaseNoGraphMatch:
    async def search_similar_chunks(self, embedding, config_id, dim, top_k):
        return [
            {"id": "11111111-1111-1111-1111-111111111111", "chunk_text": "some chunk",
             "context_text": None, "page_number": 3, "filename": "PFR1.pdf", "distance": 0.1},
        ]

    async def find_entities(self, pairs):
        return []


@pytest.mark.asyncio
async def test_answer_question_graph_augmented_merges_in_entity_matched_chunk():
    result = await answer_question(
        "For Data Counter 9017, what were the values?",
        FakeDatabaseWithGraphMatch(),
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=5,
        config=TEST_CONFIG,
        graph_augmented=True,
        ner=FakeNER([FakeEntity("DATA_COUNTER_ID", "9017")]),
    )

    chunk_ids = {s.chunk_id for s in result.sources}
    assert "22222222-2222-2222-2222-222222222222" in chunk_ids
    # Lower distance (graph-matched chunk) ranks first.
    assert result.sources[0].chunk_id == "22222222-2222-2222-2222-222222222222"


@pytest.mark.asyncio
async def test_answer_question_graph_augmented_no_entities_matches_plain_behavior():
    result = await answer_question(
        "what happened?",
        FakeDatabaseNoGraphMatch(),
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=5,
        config=TEST_CONFIG,
        graph_augmented=True,
        ner=FakeNER([]),
    )

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_answer_question_graph_augmented_false_ignores_ner():
    result = await answer_question(
        "For Data Counter 9017, what were the values?",
        FakeDatabaseWithGraphMatch(),
        FakeEmbeddingProvider(),
        FakeChatProvider(),
        top_k=5,
        config=TEST_CONFIG,
        graph_augmented=False,
    )

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "11111111-1111-1111-1111-111111111111"
