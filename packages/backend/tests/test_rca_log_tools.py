from app.services.rca_log_tools import get_backend_logs


def _write_log(path):
    path.write_text(
        "2026-09-17 10:00:00 INFO uvicorn.access: GET /api/health\n"
        "2026-09-17 10:05:00 WARNING app.services.reval_run: reval question 1: boom\n"
        "2026-09-17 10:10:00 INFO uvicorn.access: GET /api/eval/history\n",
        encoding="utf-8",
    )


def test_get_backend_logs_filters_by_time_range(tmp_path):
    log_file = tmp_path / "backend.log"
    _write_log(log_file)

    lines = get_backend_logs(str(log_file), "2026-09-17 10:01:00", "2026-09-17 10:11:00")

    assert len(lines) == 2
    assert "reval question 1: boom" in lines[0]
    assert "GET /api/eval/history" in lines[1]


def test_get_backend_logs_filters_by_grep(tmp_path):
    log_file = tmp_path / "backend.log"
    _write_log(log_file)

    lines = get_backend_logs(
        str(log_file), "2026-09-17 00:00:00", "2026-09-18 00:00:00", grep="WARNING",
    )

    assert len(lines) == 1
    assert "boom" in lines[0]


def test_get_backend_logs_missing_file_returns_empty(tmp_path):
    lines = get_backend_logs(str(tmp_path / "missing.log"), "2026-09-17 00:00:00", "2026-09-18 00:00:00")
    assert lines == []


def test_get_backend_logs_caps_at_500_lines(tmp_path):
    log_file = tmp_path / "backend.log"
    log_file.write_text(
        "".join(f"2026-09-17 10:00:{i:02d} INFO line {i}\n" for i in range(600)),
        encoding="utf-8",
    )

    lines = get_backend_logs(str(log_file), "2026-09-17 00:00:00", "2026-09-18 00:00:00")

    assert len(lines) == 500
