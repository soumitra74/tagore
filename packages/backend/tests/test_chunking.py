import json
from pathlib import Path

import pytest
import tiktoken

from app.services.chunking import (
    Chunk,
    chunk_document_structure_aware,
    chunk_fixed_token,
    chunk_page,
    chunk_page_text,
    chunk_recursive_semantic,
    chunk_sentence_window,
    extract_json_pages,
    extract_pages,
    _normalize_page_text,
    _strip_null_bytes,
)

CORPUS_DIR = Path(__file__).resolve().parents[3]


def test_chunk_short_text_returns_single_chunk():
    chunks = chunk_page_text(page_number=1, text="short text", chunk_size=1000, overlap=200)
    assert chunks == [Chunk(page_number=1, text="short text")]


def test_chunk_long_text_splits_with_overlap():
    text = "a" * 2500
    chunks = chunk_page_text(page_number=3, text=text, chunk_size=1000, overlap=200)
    assert len(chunks) == 3
    assert all(c.page_number == 3 for c in chunks)
    assert chunks[0].text == text[0:1000]
    assert chunks[1].text == text[800:1800]
    assert chunks[2].text == text[1600:2500]


def test_chunk_empty_text_returns_no_chunks():
    assert chunk_page_text(page_number=1, text="", chunk_size=1000, overlap=200) == []


def test_fixed_token_short_text_returns_single_chunk():
    chunks = chunk_fixed_token(page_number=1, text="short text", chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "short text"


def test_fixed_token_empty_text_returns_no_chunks():
    assert chunk_fixed_token(page_number=1, text="", chunk_size=512, overlap=0) == []


def test_fixed_token_long_text_splits_by_token_count_with_overlap():
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "the quick brown fox jumps over the lazy dog. " * 40
    tokens = encoding.encode(text)
    assert len(tokens) > 200  # sanity: long enough to require multiple chunks

    chunks = chunk_fixed_token(page_number=5, text=text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(c.page_number == 5 for c in chunks)
    # Every chunk decodes to at most chunk_size tokens.
    assert all(len(encoding.encode(c.text)) <= 100 for c in chunks)
    # First chunk covers exactly the first 100 tokens.
    assert encoding.encode(chunks[0].text) == tokens[:100]
    # Consecutive chunks advance by (chunk_size - overlap) tokens.
    assert encoding.encode(chunks[1].text) == tokens[80:180]


def test_recursive_semantic_short_text_returns_single_chunk():
    chunks = chunk_recursive_semantic(page_number=1, text="short text", chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "short text"


def test_recursive_semantic_empty_text_returns_no_chunks():
    assert chunk_recursive_semantic(page_number=1, text="", chunk_size=512, overlap=0) == []


def test_recursive_semantic_splits_on_paragraph_boundaries_before_hard_limit():
    encoding = tiktoken.get_encoding("cl100k_base")
    para_a = "Alpha section discusses the telemetry pipeline."
    para_b = "Beta section covers the anomaly detection thresholds."
    text = para_a + "\n\n" + para_b
    chunk_size = max(len(encoding.encode(para_a)), len(encoding.encode(para_b))) + 2
    assert len(encoding.encode(text)) > chunk_size  # sanity: combined needs a split

    chunks = chunk_recursive_semantic(page_number=2, text=text, chunk_size=chunk_size, overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text.strip() == para_a
    assert chunks[1].text.strip() == para_b


def test_recursive_semantic_never_exceeds_chunk_size_even_without_boundaries():
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "a" * 5000  # one giant run-on "word" with no natural split points

    chunks = chunk_recursive_semantic(page_number=3, text=text, chunk_size=50, overlap=0)

    assert len(chunks) > 1
    assert all(len(encoding.encode(c.text)) <= 50 for c in chunks)


def test_recursive_semantic_overlap_repeats_trailing_content_in_next_chunk():
    sentences = " ".join(f"This is sentence number {i}." for i in range(30))

    chunks = chunk_recursive_semantic(page_number=4, text=sentences, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    tail_sentence = chunks[0].text.strip().rstrip(".").split(". ")[-1] + "."
    assert tail_sentence in chunks[1].text


def test_document_structure_aware_short_text_returns_single_chunk():
    chunks = chunk_document_structure_aware(page_number=1, text="short text", chunk_size=1000, overlap=0)
    assert chunks == [Chunk(page_number=1, text="short text")]


def test_document_structure_aware_empty_text_returns_no_chunks():
    assert chunk_document_structure_aware(page_number=1, text="", chunk_size=1000, overlap=0) == []


def test_document_structure_aware_splits_on_paragraph_boundaries_when_no_headings():
    para_a = "Alpha section discusses the telemetry pipeline."
    para_b = "Beta section covers the anomaly detection thresholds."
    text = para_a + "\n\n" + para_b

    chunks = chunk_document_structure_aware(page_number=2, text=text, chunk_size=1000, overlap=0)

    assert len(chunks) == 2
    assert chunks[0] == Chunk(page_number=2, text=para_a)
    assert chunks[1] == Chunk(page_number=2, text=para_b)


def test_document_structure_aware_groups_paragraphs_under_their_heading():
    text = (
        "ACKNOWLEDGMENTS\n\n"
        "Thanks to the flight test team for their support.\n\n"
        "Additional thanks to the data processing group.\n\n"
        "LIST OF TABLES\n\n"
        "Table 1. UH-60A Rotor Parameters."
    )

    chunks = chunk_document_structure_aware(page_number=4, text=text, chunk_size=1000, overlap=0)

    assert len(chunks) == 2
    assert chunks[0].page_number == 4
    assert chunks[0].text.startswith("ACKNOWLEDGMENTS")
    assert "Thanks to the flight test team" in chunks[0].text
    assert "Additional thanks to the data processing group" in chunks[0].text
    assert "LIST OF TABLES" not in chunks[0].text
    assert chunks[1].text.startswith("LIST OF TABLES")
    assert "Table 1. UH-60A Rotor Parameters." in chunks[1].text


def test_document_structure_aware_keeps_preamble_before_first_heading_as_its_own_chunk():
    text = "Some preamble text with no heading above it.\n\nINTRODUCTION\n\nBody of the introduction."

    chunks = chunk_document_structure_aware(page_number=1, text=text, chunk_size=1000, overlap=0)

    assert len(chunks) == 2
    assert chunks[0].text == "Some preamble text with no heading above it."
    assert chunks[1].text.startswith("INTRODUCTION")


def test_document_structure_aware_ignores_chunk_size_and_overlap():
    # A section grouped under a heading stays a single chunk even when
    # chunk_size is tiny -- structure drives the split, not token count.
    text = (
        "INTRODUCTION\n\n"
        + ("This sentence is part of a long introduction section. " * 30).strip()
    )

    chunks = chunk_document_structure_aware(page_number=1, text=text, chunk_size=10, overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("INTRODUCTION")


def test_chunk_page_dispatches_to_document_structure_aware():
    text = "ACKNOWLEDGMENTS\n\nThanks all around.\n\nLIST OF TABLES\n\nTable 1. Something."
    assert chunk_page("document_structure_aware", 3, text, 1000, 0) == chunk_document_structure_aware(
        3, text, 1000, 0
    )


def test_sentence_window_empty_text_returns_no_chunks():
    assert chunk_sentence_window(page_number=1, text="", chunk_size=1, overlap=0) == []


def test_sentence_window_single_sentence_has_itself_as_context():
    chunks = chunk_sentence_window(page_number=1, text="Only one sentence here.", chunk_size=2, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."
    assert chunks[0].context_text == "Only one sentence here."


def test_sentence_window_one_chunk_per_sentence_embeds_the_sentence_alone():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_sentence_window(page_number=2, text=text, chunk_size=1, overlap=0)

    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["First sentence.", "Second sentence.", "Third sentence."]
    assert all(c.page_number == 2 for c in chunks)


def test_sentence_window_middle_sentence_gets_neighbors_on_both_sides():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_sentence_window(page_number=1, text=text, chunk_size=1, overlap=0)

    assert chunks[1].text == "Second sentence."
    assert chunks[1].context_text == "First sentence. Second sentence. Third sentence."


def test_sentence_window_edge_sentences_clamp_instead_of_erroring():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_sentence_window(page_number=1, text=text, chunk_size=1, overlap=0)

    # No sentence before the first -- context is just itself + the next one.
    assert chunks[0].context_text == "First sentence. Second sentence."
    # No sentence after the last -- context is the previous one + itself.
    assert chunks[2].context_text == "Second sentence. Third sentence."


def test_sentence_window_zero_context_size_means_no_padding():
    text = "First sentence. Second sentence."
    chunks = chunk_sentence_window(page_number=1, text=text, chunk_size=0, overlap=0)

    assert chunks[0].context_text == "First sentence."
    assert chunks[1].context_text == "Second sentence."


def test_sentence_window_larger_window_pulls_in_more_neighbors():
    text = "One. Two. Three. Four. Five."
    chunks = chunk_sentence_window(page_number=1, text=text, chunk_size=2, overlap=0)

    # "Three" (index 2) gets 2 neighbors each side -- the whole text.
    assert chunks[2].text == "Three."
    assert chunks[2].context_text == "One. Two. Three. Four. Five."


def test_sentence_window_does_not_cross_paragraph_boundaries():
    # Two separate paragraphs (blank-line separated) -- the last sentence of
    # paragraph A must not pull in the first sentence of paragraph B as
    # "surrounding context", and vice versa, even though they're adjacent
    # sentences in the raw page text.
    text = (
        "Alpha one. Alpha two.\n\n"
        "Beta one. Beta two."
    )
    chunks = chunk_sentence_window(page_number=1, text=text, chunk_size=2, overlap=0)

    assert [c.text for c in chunks] == ["Alpha one.", "Alpha two.", "Beta one.", "Beta two."]
    # "Alpha two." (last of paragraph A) should only pull in "Alpha one.",
    # never "Beta one." from the next paragraph.
    alpha_two = chunks[1]
    assert alpha_two.text == "Alpha two."
    assert alpha_two.context_text == "Alpha one. Alpha two."
    # "Beta one." (first of paragraph B) should only pull in "Beta two.",
    # never "Alpha two." from the previous paragraph.
    beta_one = chunks[2]
    assert beta_one.text == "Beta one."
    assert beta_one.context_text == "Beta one. Beta two."


def test_chunk_page_dispatches_to_sentence_window():
    text = "First sentence. Second sentence. Third sentence."
    assert chunk_page("sentence_window", 4, text, 1, 0) == chunk_sentence_window(4, text, 1, 0)


def test_chunk_page_dispatches_to_recursive_semantic():
    text = "Alpha section. " * 20 + "\n\n" + "Beta section. " * 20
    assert chunk_page("recursive_semantic", 2, text, 40, 5) == chunk_recursive_semantic(2, text, 40, 5)


def test_chunk_page_dispatches_to_sliding_window():
    text = "a" * 1500
    assert chunk_page("sliding_window", 1, text, 1000, 200) == chunk_page_text(1, text, 1000, 200)


def test_chunk_page_dispatches_to_fixed_token():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    assert chunk_page("fixed_token", 5, text, 100, 20) == chunk_fixed_token(5, text, 100, 20)


def test_extract_json_pages_splits_poems_on_page_breaks():
    raw = json.dumps(
        [
            {
                "poem_number": "আকাশপ্রদীপ",
                "collection": "আকাশপ্রদীপ",
                "url": "http://example/1",
                "pages": 1,
                "content": "         গোধূলিতে নামল আঁধার,\n                        ফুরিয়ে গেল বেলা।",
            },
            {
                "poem_number": "যাত্রাপথ",
                "collection": "আকাশপ্রদীপ",
                "url": "http://example/2",
                "pages": 2,
                "content": "প্রথম পাতা।\n---PAGE BREAK---\nদ্বিতীয় পাতা।",
            },
        ],
        ensure_ascii=False,
    ).encode()

    pages = extract_json_pages(raw)

    assert [number for number, _ in pages] == [1, 2, 3]
    assert pages[0][1] == "আকাশপ্রদীপ\nগোধূলিতে নামল আঁধার,\nফুরিয়ে গেল বেলা।"
    assert pages[1][1] == "যাত্রাপথ\nপ্রথম পাতা।"
    assert pages[2][1] == "যাত্রাপথ\nদ্বিতীয় পাতা।"
    assert all("---PAGE BREAK---" not in text for _, text in pages)


def test_extract_json_pages_skips_empty_page_and_cleans_pdf_artifacts():
    raw = json.dumps(
        [
            {
                "poem_number": "নোট",
                "pages": 3,
                "content": "শিরোনাম .......... ৫\nO\x00cer\n---PAGE BREAK---\n   \n---PAGE BREAK---\nশেষ পাতা।",
            }
        ]
    ).encode()

    pages = extract_json_pages(raw)

    assert [number for number, _ in pages] == [1, 2]
    assert "....." not in pages[0][1]
    assert "\x00" not in pages[0][1]
    assert pages[0][1].startswith("নোট\nশিরোনাম")
    assert pages[1][1] == "নোট\nশেষ পাতা।"


def test_extract_json_pages_rejects_non_list():
    with pytest.raises(ValueError, match="list of poems"):
        extract_json_pages(b'{"poem_number": "x"}')


def test_extract_pages_reads_corpus_pdf():
    pdf_path = CORPUS_DIR / "corpus" / "NASA_TelemetryAnalysis_PFR1_First23Pages.pdf"
    pages = extract_pages(pdf_path.read_bytes())
    assert len(pages) > 0
    # Page numbers are 1-indexed and sequential.
    assert [p[0] for p in pages] == list(range(1, len(pages) + 1))
    # Every page yields a string (possibly empty if the page is image-only).
    assert all(isinstance(p[1], str) for p in pages)


def test_normalize_page_text_collapses_dot_leaders():
    text = (
        "..................................... 1 \n"
        "Description of Aircraft and Rotor System "
        "................................................................................ 2 \n"
        "UH-60A Airloads"
    )
    normalized = _normalize_page_text(text)
    assert "." * 5 not in normalized
    assert "Description of Aircraft and Rotor System" in normalized
    assert "UH-60A Airloads" in normalized


def test_normalize_page_text_leaves_ordinary_sentences_and_ellipses_alone():
    assert _normalize_page_text("A short sentence.") == "A short sentence."
    assert _normalize_page_text("Wait... what happened?") == "Wait... what happened?"
    assert _normalize_page_text("e.g. a.b.c.") == "e.g. a.b.c."


def test_strip_null_bytes_removes_nul_from_misdecoded_ligatures():
    assert _strip_null_bytes("O\x00cer") == "Ocer"
    assert _strip_null_bytes("no nul here") == "no nul here"
    assert _strip_null_bytes("\x00\x00") == ""


def test_normalize_page_text_also_strips_null_bytes():
    assert "\x00" not in _normalize_page_text("O\x00cer Information")
