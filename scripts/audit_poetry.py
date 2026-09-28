#!/usr/bin/env python3
"""
Compare scraped poem JSON against the live page's #kobita innerText.

On mismatch, write a text diff plus screenshots of the live poem and the
saved text, and append a row to failures.jsonl.
"""

import argparse
import html
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

from rabindra_scraper import RabidraScraper, RabindraScraperConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PAGE_BREAK = "\n---PAGE BREAK---\n"
SKIP_JSON = {"summary.json"}


def normalize_poem_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def poem_lines(text: str) -> List[str]:
    """Non-empty lines after browser-like whitespace normalize.

    Chromium innerText inserts a blank line between <p> tags, while the
    scraper keeps one newline. Comparing only content lines ignores that.
    """
    return [line for line in normalize_poem_text(text).split("\n") if line.strip()]


def first_mismatch(saved: str, live: str) -> Optional[Dict]:
    saved_lines = poem_lines(saved)
    live_lines = poem_lines(live)
    for index in range(max(len(saved_lines), len(live_lines))):
        saved_line = saved_lines[index] if index < len(saved_lines) else None
        live_line = live_lines[index] if index < len(live_lines) else None
        if saved_line != live_line:
            return {"line": index + 1, "saved": saved_line, "live": live_line}
    return None


def split_pages(content: str) -> List[str]:
    if not content:
        return [""]
    return content.split(PAGE_BREAK)


def load_scraped_poems(output_dir: Path) -> List[Dict]:
    poems = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in SKIP_JSON:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            poems.extend(data)
    return poems


def artifact_stem(poem: Dict, page_index: int) -> str:
    match = re.search(r"/node/(\d+)", poem.get("url", ""))
    node = match.group(1) if match else "unknown"
    return f"{node}_p{page_index}"


def write_diff(path: Path, saved: str, live: str, diff: Dict) -> None:
    path.write_text(
        "\n".join([
            f"first differing content line: {diff['line']}",
            f"saved: {diff['saved']!r}",
            f"live:  {diff['live']!r}",
            "",
            "=== SAVED ===",
            saved,
            "",
            "=== LIVE innerText ===",
            live,
            "",
        ]),
        encoding="utf-8",
    )


class LiveKobita:
    """Playwright session for #kobita innerText and screenshots."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self.page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self.page = self._browser.new_page()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        return False

    def open(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self.page.locator('div[id^="kobita"]').first.wait_for(state="visible", timeout=15000)

    def inner_text(self) -> str:
        return self.page.evaluate(
            """() => {
                const nodes = [...document.querySelectorAll('div[id^="kobita"]')];
                const seen = new Set();
                const parts = [];
                for (const node of nodes) {
                    if (!node.id || seen.has(node.id)) continue;
                    seen.add(node.id);
                    const text = node.innerText;
                    if (text && text.trim()) parts.push(text);
                }
                return parts.join('\\n');
            }"""
        )

    def next_page_url(self, current_url: str) -> Optional[str]:
        link = self.page.locator("a").filter(has_text=re.compile(r"পরবর্তী|next", re.I))
        if link.count() == 0:
            return None
        href = link.first.get_attribute("href")
        if not href:
            return None
        return urljoin(current_url, href)

    def screenshot_kobita(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        locators = self.page.locator('div[id^="kobita"]')
        if locators.count() == 1:
            locators.first.screenshot(path=str(path))
            return
        box = self.page.evaluate(
            """() => {
                const nodes = [...document.querySelectorAll('div[id^="kobita"]')];
                const seen = new Set();
                let x1 = Infinity, y1 = Infinity, x2 = 0, y2 = 0, found = false;
                for (const node of nodes) {
                    if (!node.id || seen.has(node.id)) continue;
                    seen.add(node.id);
                    const r = node.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) continue;
                    found = true;
                    x1 = Math.min(x1, r.x);
                    y1 = Math.min(y1, r.y);
                    x2 = Math.max(x2, r.x + r.width);
                    y2 = Math.max(y2, r.y + r.height);
                }
                if (!found) return null;
                return {x: x1, y: y1, width: x2 - x1, height: y2 - y1};
            }"""
        )
        if box:
            self.page.screenshot(path=str(path), clip=box)
        else:
            locators.first.screenshot(path=str(path))

    def screenshot_saved_text(self, text: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        markup = (
            "<!doctype html><meta charset='utf-8'>"
            "<pre style=\"margin:0;padding:16px;font:16px/1.6 'Noto Sans Bengali',"
            "Vrinda,SolaimanLipi,sans-serif;white-space:pre-wrap;\">"
            f"{html.escape(text)}</pre>"
        )
        self.page.set_content(markup)
        self.page.locator("pre").screenshot(path=str(path))


def collect_live_pages(live: LiveKobita, start_url: str) -> List[Dict]:
    pages = []
    url = start_url
    seen = set()
    for _ in range(10):
        if not url or url in seen:
            break
        seen.add(url)
        live.open(url)
        pages.append({"url": url, "text": live.inner_text()})
        url = live.next_page_url(url)
    return pages


def audit_poems(
    output_dir: Path,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    collection: Optional[str] = None,
    delay: float = RabindraScraperConfig.DELAY_BETWEEN_REQUESTS,
    screenshots: bool = True,
) -> Dict:
    poems = load_scraped_poems(output_dir)
    if collection:
        poems = [p for p in poems if collection.lower() in (p.get("collection") or "").lower()]
    if sample and len(poems) > sample:
        poems = random.sample(poems, sample)
    if limit:
        poems = poems[:limit]

    screenshot_dir = output_dir / "audit_screenshots"
    scraper = RabidraScraper(output_dir=str(output_dir))
    results = []

    with LiveKobita() as live:
        for poem in poems:
            result = audit_one(live, scraper, poem, screenshot_dir, screenshots)
            results.append(result)
            time.sleep(delay)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checked": len(results),
        "matched": sum(1 for r in results if r["status"] == "match"),
        "mismatched": sum(1 for r in results if r["status"] == "mismatch"),
        "page_count_mismatch": sum(1 for r in results if not r.get("page_count_ok", True)),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results,
    }
    report_path = output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Audit %s/%s matched. Report: %s",
        report["matched"],
        report["checked"],
        report_path,
    )
    if scraper.failure_count:
        logger.warning("Recorded %s audit failures in %s", scraper.failure_count, scraper.failures_path)
    return report


def audit_one(
    live: LiveKobita,
    scraper: RabidraScraper,
    poem: Dict,
    screenshot_dir: Path,
    screenshots: bool,
) -> Dict:
    url = poem.get("url", "")
    title = poem.get("poem_number", "")
    collection = poem.get("collection", "")
    logger.info("Audit %s (%s)", title, url)

    try:
        live_pages = collect_live_pages(live, url)
    except Exception as exc:
        scraper._record_failure(
            reason="audit_fetch_failed",
            collection=collection,
            poem=title,
            url=url,
            detail=str(exc),
        )
        return {"url": url, "poem": title, "status": "error", "detail": str(exc)}

    if not live_pages:
        scraper._record_failure(
            reason="audit_no_kobita",
            collection=collection,
            poem=title,
            url=url,
        )
        return {"url": url, "poem": title, "status": "error", "detail": "no live pages"}

    saved_pages = split_pages(poem.get("content") or "")
    status = "match"
    diffs = []

    saved_page_count = poem.get("pages")
    page_count_wrong = len(live_pages) != len(saved_pages) or (
        saved_page_count and saved_page_count != len(live_pages)
    )
    if page_count_wrong:
        status = "page_count_mismatch"
        scraper._record_failure(
            reason="audit_page_count",
            collection=collection,
            poem=title,
            url=url,
            detail=f"saved_pages={len(saved_pages)} saved.pages={saved_page_count} live_pages={len(live_pages)}",
        )

    for index in range(min(len(live_pages), len(saved_pages))):
        live_page = live_pages[index]
        saved_text = saved_pages[index]
        diff = first_mismatch(saved_text, live_page["text"])
        if not diff:
            continue
        status = "mismatch"
        stem = artifact_stem(poem, index + 1)
        diff_path = screenshot_dir / f"{stem}_diff.txt"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        write_diff(diff_path, saved_text, live_page["text"], diff)
        artifacts = {"diff": str(diff_path)}
        if screenshots:
            live_png = screenshot_dir / f"{stem}_live.png"
            saved_png = screenshot_dir / f"{stem}_saved.png"
            try:
                live.open(live_page["url"])
                live.screenshot_kobita(live_png)
                live.screenshot_saved_text(saved_text, saved_png)
                artifacts["live_screenshot"] = str(live_png)
                artifacts["saved_screenshot"] = str(saved_png)
            except Exception as exc:
                logger.warning("Screenshot failed for %s: %s", live_page["url"], exc)
                artifacts["screenshot_error"] = str(exc)
        scraper._record_failure(
            reason="audit_mismatch",
            collection=collection,
            poem=title,
            url=live_page["url"],
            detail=json.dumps({"line": diff["line"], **artifacts}, ensure_ascii=False),
        )
        diffs.append({"page": index + 1, **diff, **artifacts})

    if page_count_wrong and screenshots:
        for index, live_page in enumerate(live_pages):
            if index < len(saved_pages):
                continue
            stem = artifact_stem(poem, index + 1)
            live_png = screenshot_dir / f"{stem}_live.png"
            try:
                live.open(live_page["url"])
                live.screenshot_kobita(live_png)
                diffs.append({
                    "page": index + 1,
                    "note": "live page has no matching saved page",
                    "live_screenshot": str(live_png),
                })
            except Exception as exc:
                logger.warning("Screenshot failed for extra page %s: %s", live_page["url"], exc)

    return {
        "url": url,
        "poem": title,
        "collection": collection,
        "status": status,
        "saved_pages": len(saved_pages),
        "live_pages": len(live_pages),
        "page_count_ok": not page_count_wrong,
        "diffs": diffs,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit scraped poems against live #kobita innerText")
    parser.add_argument("--input", default=RabindraScraperConfig.OUTPUT_DIR, help="Scraped JSON directory")
    parser.add_argument("--collection", help="Collection name (partial match OK)")
    parser.add_argument("--limit", type=int, help="Audit only the first N poems")
    parser.add_argument("--sample", type=int, help="Randomly audit this many poems")
    parser.add_argument(
        "--delay",
        type=float,
        default=RabindraScraperConfig.DELAY_BETWEEN_REQUESTS,
        help="Delay between poems",
    )
    parser.add_argument("--no-screenshot", action="store_true", help="Skip PNG captures on mismatch")
    args = parser.parse_args()

    report = audit_poems(
        output_dir=Path(args.input),
        limit=args.limit,
        sample=args.sample,
        collection=args.collection,
        delay=args.delay,
        screenshots=not args.no_screenshot,
    )
    if report["mismatched"] or report["errors"] or report["page_count_mismatch"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
