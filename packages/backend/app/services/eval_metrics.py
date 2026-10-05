import math


def compute_hit(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]]
) -> bool:
    return any(page in golden_pages for page in retrieved_pages)


def compute_mrr(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]]
) -> float:
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in golden_pages:
            return 1.0 / rank
    return 0.0


def compute_citation_accuracy(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]]
) -> float:
    if not retrieved_pages:
        return 0.0
    matches = sum(1 for page in retrieved_pages if page in golden_pages)
    return matches / len(retrieved_pages)


def compute_citation_recall(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]]
) -> float:
    """Precision/recall counterpart to compute_citation_accuracy: what
    fraction of the golden pages were actually retrieved. Iterates over the
    golden set (not retrieved_pages) so a golden page retrieved via more
    than one chunk is only counted once."""
    if not golden_pages:
        return 1.0
    retrieved_set = set(retrieved_pages)
    matches = sum(1 for page in golden_pages if page in retrieved_set)
    return matches / len(golden_pages)


def compute_ndcg_at_k(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]], k: int
) -> float:
    """nDCG@k with binary relevance (the golden data marks gold pages, with no
    grades): DCG sums 1/log2(rank+1) for each gold page over the first k
    ranked pages, normalized by the ideal DCG (all min(#gold, k) gold pages
    ranked first). A gold page only scores at its FIRST appearance -- later
    chunks from the same page score 0 -- so DCG can never exceed the ideal
    (same no-double-counting rule as compute_citation_recall)."""
    if not golden_pages:
        return 1.0
    seen: set[tuple[str, int]] = set()
    dcg = 0.0
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in golden_pages and page not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(page)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(golden_pages), k) + 1))
    return dcg / ideal


def compute_recall_at_k(
    retrieved_pages: list[tuple[str, int]], golden_pages: set[tuple[str, int]], k: int
) -> float:
    """compute_citation_recall restricted to the first k retrieved pages --
    the same fraction-of-golden-pages-found computation, just capped to a
    specific candidate-pool depth. Deliberately takes retrieved_pages as a
    RANKED list (not pre-truncated) so the caller can fetch one large
    candidate pool once and compute recall at every configured k from the
    same list, rather than re-fetching per k."""
    return compute_citation_recall(retrieved_pages[:k], golden_pages)
