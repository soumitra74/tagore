import difflib
import time
from dataclasses import dataclass, field

NO_CONTEXT_ANSWER = "I couldn't find relevant information in the ingested documents."

# Near-duplicate chunks are common in this corpus (e.g. PFR1/PFR2 share
# boilerplate methodology paragraphs almost verbatim). Over-fetch candidates
# beyond top_k so there's headroom to drop duplicates and still return
# top_k distinct chunks.
DEDUP_SIMILARITY_THRESHOLD = 0.9
OVERFETCH_MULTIPLIER = 3
MAX_FETCH = 30


@dataclass
class Source:
    document: str
    page: int
    snippet: str
    chunk_id: str


@dataclass
class ChatResult:
    answer: str
    sources: list[Source]
    context_chunks: list[str] = field(default_factory=list)
    generation_input_tokens: int | None = None
    generation_output_tokens: int | None = None
    rerank_latency_ms: int | None = None


async def rerank_rows(reranker, question: str, rows: list, candidates: int) -> list:
    """Reorders the first `candidates` rows by cross-encoder score (best
    first); rows beyond that keep their retrieval order after them."""
    head, tail = rows[:candidates], rows[candidates:]
    if not head:
        return rows
    scores = await reranker.rerank(question, [_display_text(r) for r in head])
    order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
    return [head[i] for i in order] + tail


def _is_near_duplicate(text: str, kept_texts: list[str]) -> bool:
    return any(
        difflib.SequenceMatcher(None, text, kept).ratio() >= DEDUP_SIMILARITY_THRESHOLD
        for kept in kept_texts
    )


def _display_text(row) -> str:
    """What the generator (and the faithfulness/info_recall judges, which
    score against the same context) actually sees for a chunk. Only
    sentence_window rows have a non-null context_text (the sentence padded
    with surrounding sentences) -- everything else falls back to the
    embedded chunk_text unchanged."""
    return row["context_text"] or row["chunk_text"]


def _dedupe_rows(rows, top_k: int) -> list:
    kept = []
    kept_texts: list[str] = []
    for row in rows:
        text = row["chunk_text"]
        if _is_near_duplicate(text, kept_texts):
            continue
        kept.append(row)
        kept_texts.append(text)
        if len(kept) >= top_k:
            break
    return kept


async def retrieve_candidates(
    question: str, db, embedding_provider, config, fetch_k: int,
    graph_augmented: bool = False, ner=None,
) -> list:
    """Embeds the question once and fetches a candidate pool, optionally
    merged with graph-augmented rows. Split out of answer_question so a
    single retrieval can be reused for more than one purpose -- eval_run.py
    shares this same pool between generation and its ranking diagnostics
    instead of fetching (and, worse, reranking) it twice per question."""
    question_embedding = await embedding_provider.embed_query(question)
    candidate_rows = await db.search_similar_chunks(
        question_embedding, config.id, config.embedding_dim, fetch_k
    )

    if graph_augmented and ner is not None:
        extracted = await ner.extract(question)
        pairs = [(e.entity_type, e.normalized_text) for e in extracted]
        entity_ids = await db.find_entities(pairs) if pairs else []
        if entity_ids:
            graph_rows = await db.get_chunks_mentioning_entities(
                config.id, config.embedding_dim, entity_ids, question_embedding, fetch_k
            )
            merged = {row["id"]: row for row in candidate_rows}
            for row in graph_rows:
                merged.setdefault(row["id"], row)
            candidate_rows = sorted(merged.values(), key=lambda r: r["distance"])

    return candidate_rows


async def generate_from_candidates(
    question: str, candidate_rows: list, chat_provider, top_k: int,
) -> ChatResult:
    """Dedupes an already-retrieved (and optionally reranked) candidate pool
    down to top_k and generates the answer. Split out of answer_question so
    eval_run.py can call this directly with a pool it already retrieved and
    reranked itself, rather than making answer_question repeat that work."""
    rows = _dedupe_rows(candidate_rows, top_k)

    if not rows:
        return ChatResult(answer=NO_CONTEXT_ANSWER, sources=[])

    context_chunks = [_display_text(row) for row in rows]
    labeled_chunks = [
        f"[Source {i} -- {row['filename']}, page {row['page_number']}]\n{_display_text(row)}"
        for i, row in enumerate(rows, start=1)
    ]
    llm_response = await chat_provider.generate(question, labeled_chunks)
    sources = [
        Source(
            document=row["filename"], page=row["page_number"],
            snippet=row["chunk_text"][:200], chunk_id=row["id"],
        )
        for row in rows
    ]
    return ChatResult(
        answer=llm_response.text,
        sources=sources,
        context_chunks=context_chunks,
        generation_input_tokens=llm_response.input_tokens,
        generation_output_tokens=llm_response.output_tokens,
    )


async def answer_question(
    question: str, db, embedding_provider, chat_provider, top_k: int, config,
    graph_augmented: bool = False, ner=None,
    reranker=None, rerank_candidates: int | None = None,
) -> ChatResult:
    fetch_k = min(top_k * OVERFETCH_MULTIPLIER, MAX_FETCH)
    if reranker is not None and rerank_candidates:
        fetch_k = max(fetch_k, rerank_candidates)
    candidate_rows = await retrieve_candidates(
        question, db, embedding_provider, config, fetch_k, graph_augmented, ner
    )

    rerank_latency_ms = None
    if reranker is not None and rerank_candidates:
        rerank_start = time.monotonic()
        candidate_rows = await rerank_rows(
            reranker, question, candidate_rows, rerank_candidates
        )
        rerank_latency_ms = int((time.monotonic() - rerank_start) * 1000)

    result = await generate_from_candidates(question, candidate_rows, chat_provider, top_k)
    result.rerank_latency_ms = rerank_latency_ms
    return result
