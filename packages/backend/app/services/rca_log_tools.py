_TIMESTAMP_LEN = len("2026-09-17 10:00:00")
MAX_LINES = 500


def get_backend_logs(log_file: str, since: str, until: str, grep: str | None = None) -> list[str]:
    """Reads the backend's log file (written by main.py's RotatingFileHandler)
    and filters by the timestamp every line is prefixed with, plus an
    optional case-insensitive substring. Missing file (e.g. nothing logged
    yet) returns an empty list rather than raising -- a quiet backend isn't
    an error for this tool."""
    try:
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    matched = []
    for line in lines:
        timestamp = line[:_TIMESTAMP_LEN]
        if timestamp < since or timestamp > until:
            continue
        if grep and grep.lower() not in line.lower():
            continue
        matched.append(line.rstrip("\n"))
    return matched[-MAX_LINES:]
