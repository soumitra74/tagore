import json
import re
from dataclasses import dataclass
from io import BytesIO

import tiktoken
from pypdf import PdfReader

_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    page_number: int
    text: str
    # Set only by chunk_sentence_window(): the sentence padded with
    # surrounding context. text alone is what gets embedded (precise
    # single-sentence retrieval); context_text is what the generator sees
    # (see chat.py, which prefers this over text when present). None for
    # every other strategy, which behave exactly as before.
    context_text: str | None = None


# TOC/index dot leaders ("Introduction .......... 5"): 5+ repeats of a dot/bullet optionally
# separated by spaces. Collapsing these before chunking keeps embeddings from being diluted by
# filler characters and stops a TOC line's title words from scoring as a false-positive match
# for a query about that section's actual content.
_DOT_LEADER = re.compile(r"[.•…](?:\s*[.•…]){4,}")


# pypdf occasionally mis-decodes a ligature glyph (e.g. a font's "ffi" glyph) as a literal NUL
# byte instead of the letters it represents -- e.g. "Officer" extracts as "O\x00cer". Postgres's
# text type rejects NUL outright ("invalid byte sequence for encoding UTF8: 0x00"), so any PDF
# that triggers this fails ingestion entirely unless the byte is removed before storage. Losing
# the glyph (rather than guessing which letters it stood for) is an acceptable trade for a
# handful of characters out of a whole document.
def _strip_null_bytes(text: str) -> str:
    return text.replace("\x00", "")


def _normalize_page_text(text: str) -> str:
    return _DOT_LEADER.sub(" ", _strip_null_bytes(text))


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [
        (i + 1, _normalize_page_text(page.extract_text() or ""))
        for i, page in enumerate(reader.pages)
    ]


# The poetry JSON collections mark a printed page boundary inside a poem's
# `content` with this literal. It plays the same role as a PDF page object:
# everything between markers is one page for chunking and citation.
_PAGE_BREAK = "---PAGE BREAK---"


def _normalize_verse_page(text: str) -> str:
    """Same cleanup as a PDF page, then drop the print-layout indentation
    these JSON files carry on every line. Line breaks stay, because they
    are the verse; runs of blank lines collapse to a single stanza break."""
    text = _normalize_page_text(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if collapsed and not blank:
                collapsed.append("")
            blank = True
            continue
        collapsed.append(line)
        blank = False
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return "\n".join(collapsed)


def extract_json_pages(raw: bytes) -> list[tuple[int, str]]:
    """Turn one collection file into the same (page_number, text) pairs
    extract_pages() produces for a PDF.

    Each object is one poem. `content` is split on ---PAGE BREAK---, and
    the pieces are numbered sequentially through the file the way PDF pages
    are numbered through a document. `pages` is the source's own count of
    those pieces; the marker is what actually separates them. The poem
    title (`poem_number`) is kept at the top of every page so a later page
    of a long poem still names the poem it belongs to.
    """
    try:
        poems = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corpus JSON is not valid: {exc}") from exc
    if not isinstance(poems, list):
        raise ValueError("Corpus JSON must be a list of poems")

    pages: list[tuple[int, str]] = []
    for poem in poems:
        if not isinstance(poem, dict):
            raise ValueError("Each poem must be an object")
        title = str(poem.get("poem_number") or "").strip()
        content = poem.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        for segment in content.split(_PAGE_BREAK):
            body = _normalize_verse_page(segment)
            if not body:
                continue
            text = f"{title}\n{body}" if title else body
            pages.append((len(pages) + 1, text))
    return pages


def chunk_page_text(
    page_number: int, text: str, chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    if not text:
        return []
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(page_number=page_number, text=text[start:end]))
        if end == len(text):
            break
        start += step
    return chunks


def chunk_fixed_token(
    page_number: int, text: str, chunk_size: int = 512, overlap: int = 0
) -> list[Chunk]:
    if not text:
        return []
    tokens = _TOKEN_ENCODING.encode(text)
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = _TOKEN_ENCODING.decode(tokens[start:end])
        chunks.append(Chunk(page_number=page_number, text=chunk_text))
        if end == len(tokens):
            break
        start += step
    return chunks


_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _token_len(text: str) -> int:
    return len(_TOKEN_ENCODING.encode(text))


def _split_on_boundaries(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Break text into boundary-aligned pieces, only splitting a piece
    further when it doesn't already fit in chunk_size — trying the
    coarsest separator (paragraph) first and falling through to a finer
    one (line, sentence, word) only for pieces still too big. The
    separator is kept attached to the end of each piece (rather than
    discarded by str.split) so rejoining pieces reconstructs the original
    text exactly, punctuation and all."""
    if _token_len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text]
    sep, *rest = separators
    if sep not in text:
        return _split_on_boundaries(text, rest, chunk_size)
    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part
        if piece:
            pieces.extend(_split_on_boundaries(piece, rest, chunk_size))
    return pieces


def _hard_slice_by_tokens(unit: str, chunk_size: int) -> list[str]:
    tokens = _TOKEN_ENCODING.encode(unit)
    return [
        _TOKEN_ENCODING.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]


def chunk_recursive_semantic(
    page_number: int, text: str, chunk_size: int = 512, overlap: int = 0
) -> list[Chunk]:
    if not text:
        return []

    raw_units = [u for u in _split_on_boundaries(text, _RECURSIVE_SEPARATORS, chunk_size) if u.strip()]
    units: list[str] = []
    for unit in raw_units:
        if _token_len(unit) > chunk_size:
            units.extend(_hard_slice_by_tokens(unit, chunk_size))
        else:
            units.append(unit)
    if not units:
        return []

    chunk_texts: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit_len = _token_len(unit)
        if current and current_len + unit_len > chunk_size:
            chunk_texts.append("".join(current))
            overlap_units: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                prev_len = _token_len(prev)
                if overlap_len + prev_len > overlap:
                    break
                overlap_units.insert(0, prev)
                overlap_len += prev_len
            current, current_len = overlap_units, overlap_len
        current.append(unit)
        current_len += unit_len
    if current:
        chunk_texts.append("".join(current))

    return [Chunk(page_number=page_number, text=t) for t in chunk_texts]


_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
_MAX_HEADING_WORDS = 8
_TITLE_CASE_STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in",
    "nor", "of", "on", "or", "the", "to", "with",
}


def _is_title_case(line: str) -> bool:
    """Like str.istitle(), but tolerant of lowercase connector words (e.g.
    "Definition of 0-Degree Rotor Azimuth") the way real document headings
    are actually capitalized, unlike Python's strict titlecase rule."""
    words = [w for w in line.split() if any(c.isalpha() for c in w)]
    if not words:
        return False
    for i, word in enumerate(words):
        if i > 0 and word.lower() in _TITLE_CASE_STOPWORDS:
            continue
        first_letter = next((c for c in word if c.isalpha()), None)
        if first_letter is not None and not first_letter.isupper():
            return False
    return True


def _is_heading(paragraph: str) -> bool:
    """A paragraph counts as a section heading if it's a single short line,
    isn't a sentence (no trailing punctuation), and is either ALL CAPS or
    Title Case -- the two heading styles found in the target corpus (e.g.
    "ACKNOWLEDGMENTS", "Definition of 0-Degree Rotor Azimuth")."""
    lines = paragraph.strip().splitlines()
    if len(lines) != 1:
        return False
    line = lines[0].strip()
    if not line or line.endswith((".", ",", ";")):
        return False
    if len(line.split()) > _MAX_HEADING_WORDS:
        return False
    if not any(c.isalpha() for c in line):
        return False
    return line.isupper() or _is_title_case(line)


def chunk_document_structure_aware(
    page_number: int, text: str, chunk_size: int = 0, overlap: int = 0
) -> list[Chunk]:
    """Split on the page's own section/paragraph boundaries, regardless of
    resulting chunk size -- chunk_size/overlap are accepted only to match
    the shared chunk_page() signature and don't constrain the output here.
    Paragraphs are grouped under the heading that precedes them; a page
    with no detected headings falls back to one chunk per paragraph."""
    if not text or not text.strip():
        return []

    paragraphs = [p for p in _PARAGRAPH_BOUNDARY.split(text.strip()) if p.strip()]
    if not paragraphs:
        return []

    if not any(_is_heading(p) for p in paragraphs):
        return [Chunk(page_number=page_number, text=p) for p in paragraphs]

    chunk_texts: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if _is_heading(paragraph) and current:
            chunk_texts.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunk_texts.append("\n\n".join(current))

    return [Chunk(page_number=page_number, text=t) for t in chunk_texts]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def chunk_sentence_window(
    page_number: int, text: str, chunk_size: int = 2, overlap: int = 0
) -> list[Chunk]:
    """Each chunk embeds a single sentence (precise retrieval); chunk_size
    sentences of context on each side are attached separately as
    context_text, which chat.py shows the generator instead of the bare
    sentence. overlap is unused -- accepted only to match chunk_page()'s
    shared signature.

    The neighbor window is clamped at paragraph boundaries (blank lines),
    not just the page boundary: sentences are split within each paragraph
    independently, so the sentence ending one paragraph never pulls in the
    sentence starting the next as "surrounding context" -- they may be
    about entirely unrelated things."""
    if not text or not text.strip():
        return []

    paragraphs = [p for p in _PARAGRAPH_BOUNDARY.split(text.strip()) if p.strip()]
    if not paragraphs:
        return []

    window = max(chunk_size, 0)
    chunks = []
    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        for i, sentence in enumerate(sentences):
            start = max(i - window, 0)
            end = min(i + window + 1, len(sentences))
            context_text = " ".join(sentences[start:end])
            chunks.append(Chunk(page_number=page_number, text=sentence, context_text=context_text))
    return chunks


_STRATEGY_CHUNKERS = {
    "sliding_window": chunk_page_text,
    "fixed_token": chunk_fixed_token,
    "recursive_semantic": chunk_recursive_semantic,
    "document_structure_aware": chunk_document_structure_aware,
    "sentence_window": chunk_sentence_window,
}


def chunk_page(strategy: str, page_number: int, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    return _STRATEGY_CHUNKERS[strategy](page_number, text, chunk_size, overlap)
