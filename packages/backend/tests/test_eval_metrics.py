import math

import pytest

from app.services.eval_metrics import (
    compute_citation_accuracy,
    compute_citation_recall,
    compute_hit,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)

GOLDEN = {("doc.pdf", 5), ("doc.pdf", 6)}


def test_compute_hit_true_when_any_page_matches():
    assert compute_hit([("doc.pdf", 1), ("doc.pdf", 5)], GOLDEN) is True


def test_compute_hit_false_when_no_page_matches():
    assert compute_hit([("doc.pdf", 1), ("doc.pdf", 2)], GOLDEN) is False


def test_compute_hit_false_on_empty_retrieval():
    assert compute_hit([], GOLDEN) is False


def test_compute_mrr_uses_first_matching_rank():
    assert compute_mrr([("doc.pdf", 1), ("doc.pdf", 5), ("doc.pdf", 6)], GOLDEN) == 0.5


def test_compute_mrr_zero_when_no_match():
    assert compute_mrr([("doc.pdf", 1), ("doc.pdf", 2)], GOLDEN) == 0.0


def test_compute_mrr_zero_on_empty_retrieval():
    assert compute_mrr([], GOLDEN) == 0.0


def test_compute_citation_accuracy_fraction_of_matches():
    retrieved = [("doc.pdf", 5), ("doc.pdf", 1), ("doc.pdf", 6), ("doc.pdf", 2)]
    assert compute_citation_accuracy(retrieved, GOLDEN) == 0.5


def test_compute_citation_accuracy_zero_on_empty_retrieval():
    assert compute_citation_accuracy([], GOLDEN) == 0.0


def test_compute_citation_recall_fraction_of_golden_pages_matched():
    retrieved = [("doc.pdf", 5), ("doc.pdf", 1)]
    assert compute_citation_recall(retrieved, GOLDEN) == 0.5


def test_compute_citation_recall_full_when_all_golden_pages_matched():
    retrieved = [("doc.pdf", 1), ("doc.pdf", 5), ("doc.pdf", 6)]
    assert compute_citation_recall(retrieved, GOLDEN) == 1.0


def test_compute_citation_recall_zero_on_empty_retrieval():
    assert compute_citation_recall([], GOLDEN) == 0.0


def test_compute_citation_recall_does_not_double_count_duplicate_retrieved_pages():
    # Two retrieved chunks both happen to come from the same golden page --
    # recall should still count that golden page once, not inflate past it.
    retrieved = [("doc.pdf", 5), ("doc.pdf", 5)]
    assert compute_citation_recall(retrieved, GOLDEN) == 0.5


def test_compute_citation_recall_one_when_no_golden_pages():
    # Nothing to find -- vacuously complete, mirrors compute_citation_accuracy's
    # safe handling of the degenerate empty-set case.
    assert compute_citation_recall([("doc.pdf", 1)], set()) == 1.0


def test_compute_recall_at_k_restricts_to_first_k_candidates():
    # Golden page ("doc.pdf", 6) only appears at candidate position 3 -- found
    # at k=5 but not at k=2, from the SAME ranked candidate list.
    candidates = [("doc.pdf", 1), ("doc.pdf", 2), ("doc.pdf", 6), ("doc.pdf", 9), ("doc.pdf", 10)]
    assert compute_recall_at_k(candidates, GOLDEN, k=2) == 0.0
    assert compute_recall_at_k(candidates, GOLDEN, k=5) == 0.5


def test_compute_recall_at_k_k_larger_than_candidate_list():
    candidates = [("doc.pdf", 5)]
    assert compute_recall_at_k(candidates, GOLDEN, k=50) == 0.5


def test_compute_recall_at_k_one_when_no_golden_pages():
    assert compute_recall_at_k([("doc.pdf", 1)], set(), k=5) == 1.0


def test_compute_ndcg_at_k_perfect_ranking_is_one():
    assert compute_ndcg_at_k([("doc.pdf", 5), ("doc.pdf", 6), ("doc.pdf", 1)], GOLDEN, k=10) == pytest.approx(1.0)


def test_compute_ndcg_at_k_discounts_lower_ranked_gold_page():
    # One gold page only, at rank 3: DCG = 1/log2(4) = 0.5, ideal = 1/log2(2) = 1.
    golden = {("doc.pdf", 5)}
    retrieved = [("doc.pdf", 1), ("doc.pdf", 2), ("doc.pdf", 5)]
    assert compute_ndcg_at_k(retrieved, golden, k=10) == pytest.approx(0.5)


def test_compute_ndcg_at_k_rewards_finding_all_gold_pages_not_just_the_first():
    # Gold pages at ranks 1 and 3: DCG = 1 + 0.5, ideal = 1 + 1/log2(3).
    retrieved = [("doc.pdf", 5), ("doc.pdf", 1), ("doc.pdf", 6)]
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert compute_ndcg_at_k(retrieved, GOLDEN, k=10) == pytest.approx(expected)
    # MRR can't tell this apart from finding only one of the two gold pages.
    assert compute_mrr(retrieved, GOLDEN) == compute_mrr([("doc.pdf", 5), ("doc.pdf", 1)], GOLDEN)


def test_compute_ndcg_at_k_does_not_double_count_repeated_gold_page():
    # Two chunks from the same gold page must not push nDCG above 1.
    golden = {("doc.pdf", 5)}
    retrieved = [("doc.pdf", 5), ("doc.pdf", 5), ("doc.pdf", 5)]
    assert compute_ndcg_at_k(retrieved, golden, k=10) == pytest.approx(1.0)


def test_compute_ndcg_at_k_ignores_pages_beyond_k():
    retrieved = [("doc.pdf", i) for i in range(1, 11)] + [("doc.pdf", 99)]  # gold at rank 11
    golden = {("doc.pdf", 99)}
    assert compute_ndcg_at_k(retrieved, golden, k=10) == 0.0


def test_compute_ndcg_at_k_zero_when_no_match():
    assert compute_ndcg_at_k([("doc.pdf", 1), ("doc.pdf", 2)], GOLDEN, k=10) == 0.0


def test_compute_ndcg_at_k_one_when_no_golden_pages():
    assert compute_ndcg_at_k([("doc.pdf", 1)], set(), k=10) == 1.0
