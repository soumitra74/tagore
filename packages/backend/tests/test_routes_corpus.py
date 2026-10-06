from app.api.routes_corpus import _LIBRARY_SUFFIXES, _list_corpus_filenames


def test_library_listing_includes_json_and_pdf_only(tmp_path):
    (tmp_path / "গীতাঞ্জলি.json").write_text("[]", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.json").write_text("[]", encoding="utf-8")

    assert _list_corpus_filenames(str(tmp_path), suffixes=_LIBRARY_SUFFIXES) == [
        "report.pdf",
        "গীতাঞ্জলি.json",
    ]


def test_ingest_listing_includes_json_and_pdf(tmp_path):
    (tmp_path / "গীতাঞ্জলি.json").write_text("[]", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF")

    assert _list_corpus_filenames(str(tmp_path)) == ["report.pdf", "গীতাঞ্জলি.json"]


def test_missing_corpus_dir_lists_nothing(tmp_path):
    assert _list_corpus_filenames(str(tmp_path / "missing"), suffixes=_LIBRARY_SUFFIXES) == []
