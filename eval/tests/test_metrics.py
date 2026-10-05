from metrics import compute_citation_accuracy, compute_hit, compute_mrr

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
