#!/usr/bin/env python3
"""
Rabindra Rachanabali Bengali Poetry Scraper
Scrapes poems from https://rabindra-rachanabali.nltr.org
Handles multi-page poems and respects rate limits
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabindraScraperConfig:
    BASE_URL = "https://rabindra-rachanabali.nltr.org"
    POETRY_INDEX_URL = f"{BASE_URL}/node/10609"
    DELAY_BETWEEN_REQUESTS = 1.5
    TIMEOUT = 10
    OUTPUT_DIR = "rabindra_poetry"

class RabidraScraper:
    def __init__(self, output_dir: str = RabindraScraperConfig.OUTPUT_DIR):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.failure_count = 0
        self.failures_path = self.output_dir / "failures.jsonl"

    def _record_failure(self, reason: str, collection: str = "", poem: str = "",
                        url: str = "", detail: str = "") -> None:
        """Append one failure record to failures.jsonl"""
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "collection": collection,
            "poem": poem,
            "url": url,
        }
        if detail:
            record["detail"] = detail

        with open(self.failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.failure_count += 1
        logger.warning("  Recorded failure: %s %s", reason, url or poem or collection)

    def fetch_page(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Fetch and parse a page with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=RabindraScraperConfig.TIMEOUT)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to fetch {url}")
                    return None

            time.sleep(RabindraScraperConfig.DELAY_BETWEEN_REQUESTS)

    def _paragraph_text(self, element) -> str:
        """Turn one HTML block into poem text the way a browser would.

        Source files wrap long lines with real newline characters inside <p>.
        Browsers collapse those, plus other regular whitespace, to a single
        space. get_text() keeps them, which showed up as extra poem lines.
        &nbsp; is real indentation and is not collapsed.
        <br> is an intentional line break.
        """
        for br in element.find_all('br'):
            br.replace_with('\x00BR\x00')

        text = element.get_text(separator='')
        text = text.replace('\xa0', '\x00NBSP\x00')

        lines = []
        for part in text.split('\x00BR\x00'):
            part = re.sub(r'[ \t\r\n\f]+', ' ', part)
            part = part.strip(' ')
            part = part.replace('\x00NBSP\x00', ' ')
            if not part.strip():
                lines.append('')
            else:
                lines.append(part.rstrip())
        return '\n'.join(lines)

    def extract_poem_text(self, soup: BeautifulSoup) -> str:
        """Extract poem content from page, preserving line breaks and indentation"""
        # First, look for the poem in div#kobita (kobita = poem in Bengali)
        kobita_div = soup.find('div', id='kobita')
        if kobita_div:
            paragraphs = kobita_div.find_all('p')
            if paragraphs:
                return '\n'.join(self._paragraph_text(p) for p in paragraphs).strip('\n')

            return self._paragraph_text(kobita_div).strip()

        # Fallback: try other selectors
        content_div = None
        for selector in ['div.content', 'div.node-content', 'article', 'div.node']:
            content_div = soup.select_one(selector)
            if content_div:
                break

        if not content_div:
            content_div = soup.body or soup

        text_parts = []
        for element in content_div.find_all(['p', 'div', 'pre']):
            if element.get('class'):
                class_str = ' '.join(element.get('class', []))
                skip_keywords = ['nav', 'menu', 'footer', 'header', 'sidebar', 'breadcrumb', 'navigation', 'banner']
                if any(kw in class_str.lower() for kw in skip_keywords):
                    continue

            text = element.get_text(strip=True)
            if text and len(text) > 5:
                if not re.search(r'^(পৃষ্ঠা|page|visitor|disclaimer|পরবর্তী|পূর্ববর্তী)', text.lower()):
                    if text not in ['রবীন্দ্র-রচনাসমগ্র', 'Rabindra Rachanabali']:
                        text_parts.append(text)

        return '\n'.join(text_parts).strip() if text_parts else ""

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract poem title"""
        title_tag = soup.find('title')
        if title_tag:
            text = title_tag.get_text()
            return text.split('|')[0].strip()
        return "Unknown"

    def detect_pagination_info(self, soup: BeautifulSoup) -> tuple:
        """Detect if poem has multiple pages. Returns (current_page, total_pages, has_next)"""
        # Look for "পৃষ্ঠা १/२" pattern (page indicator)
        page_text = soup.find(string=lambda text: text and 'পৃষ্ঠা' in text if text else False)

        if page_text:
            # Extract page numbers from pattern like "পৃষ্ঠা १/२"
            match = re.search(r'(\d+)/(\d+)', str(page_text))
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                has_next = current < total
                logger.info(f"  Multi-page poem detected: page {current}/{total}")
                return (current, total, has_next)

        return (1, 1, False)

    def scrape_poem(self, poem_url: str, poem_num: str, collection_name: str) -> Optional[Dict]:
        """Scrape a single poem, handling multiple pages"""
        logger.info(f"  Poem {poem_num}...")

        all_text = []
        current_url = poem_url
        page_count = 0

        while current_url and page_count < 10:  # Safety limit
            page_count += 1

            soup = self.fetch_page(current_url)
            if not soup:
                self._record_failure(
                    reason="fetch_failed" if page_count == 1 else "page_fetch_failed",
                    collection=collection_name,
                    poem=poem_num,
                    url=current_url,
                    detail=f"page {page_count}",
                )
                break

            # Extract text
            text = self.extract_poem_text(soup)
            if text:
                all_text.append(text)
            else:
                self._record_failure(
                    reason="empty_page",
                    collection=collection_name,
                    poem=poem_num,
                    url=current_url,
                    detail=f"page {page_count}",
                )

            # Check pagination
            current_page, total_pages, has_next = self.detect_pagination_info(soup)

            # Try to find next page link
            next_url = None
            if has_next:
                # Try query parameter approach
                if '?' in current_url:
                    next_url = current_url.split('?')[0] + f"?page={current_page}"
                else:
                    next_url = current_url + f"?page={current_page}"

                # Also try to find explicit next link
                for link in soup.find_all('a'):
                    if any(text in link.get_text().lower() for text in ['next', 'পরবর্তী']):
                        href = link.get('href')
                        if href:
                            next_url = href if href.startswith('http') else RabindraScraperConfig.BASE_URL + href
                            break

            current_url = next_url if has_next else None

        if not all_text:
            logger.warning(f"  No text extracted for poem {poem_num}")
            self._record_failure(
                reason="empty_text",
                collection=collection_name,
                poem=poem_num,
                url=poem_url,
            )
            return None

        return {
            "poem_number": poem_num,
            "collection": collection_name,
            "url": poem_url,
            "pages": page_count,
            "content": '\n---PAGE BREAK---\n'.join(all_text) if page_count > 1 else all_text[0]
        }

    def scrape_collection(self, collection_url: str, collection_name: str,
                         limit: Optional[int] = None) -> List[Dict]:
        """Scrape poems from a collection"""
        logger.info(f"Scraping: {collection_name}")

        soup = self.fetch_page(collection_url)
        if not soup:
            self._record_failure(
                reason="collection_fetch_failed",
                collection=collection_name,
                url=collection_url,
            )
            return []

        poems = []

        # Poems are in the 3rd table (index 2) on collection pages
        # Table 1 = fullscreen buttons, Table 2 = navigation, Table 3 = actual poems
        tables = soup.find_all('table')
        links = []

        if len(tables) >= 3:
            # Use the 3rd table where actual poem links are located
            poem_table = tables[2]
            links = poem_table.find_all('a', href=re.compile(r'/node/\d+'))
        elif tables:
            # Fallback: try the last table
            links = tables[-1].find_all('a', href=re.compile(r'/node/\d+'))
        else:
            # Last resort: search entire page for poem links
            links = soup.find_all('a', href=re.compile(r'/node/\d+'))

        # Filter to poem links - accept both numeric AND titled links
        # Exclude navigation links that have subcatid parameter
        poem_links = [(link.get_text(strip=True), link.get('href'))
                     for link in links
                     if link.get_text(strip=True)  # Has text content
                     and '/node/' in link.get('href', '')
                     and 'subcatid' not in link.get('href', '')  # Exclude category navigation
                     and len(link.get_text(strip=True)) > 1]  # Exclude single-char links

        logger.info(f"  Found {len(poem_links)} poems")
        if not poem_links:
            self._record_failure(
                reason="no_poem_links",
                collection=collection_name,
                url=collection_url,
            )
            return []

        if limit:
            poem_links = poem_links[:limit]

        for poem_num, poem_href in poem_links:
            poem_url = poem_href if poem_href.startswith('http') else RabindraScraperConfig.BASE_URL + poem_href

            poem_data = self.scrape_poem(poem_url, poem_num, collection_name)
            if poem_data:
                poems.append(poem_data)

            time.sleep(RabindraScraperConfig.DELAY_BETWEEN_REQUESTS)

        # Save
        if poems:
            self.save_collection(collection_name, poems)

        return poems

    def get_all_collections(self) -> List[tuple]:
        """Get all poetry collections"""
        logger.info("Fetching index...")
        soup = self.fetch_page(RabindraScraperConfig.POETRY_INDEX_URL)
        if not soup:
            self._record_failure(
                reason="index_fetch_failed",
                url=RabindraScraperConfig.POETRY_INDEX_URL,
            )
            return []

        links = soup.find_all('a', href=re.compile(r'subcatid='))
        collections = [(link.get_text(strip=True),
                       link.get('href') if link.get('href').startswith('http')
                       else RabindraScraperConfig.BASE_URL + link.get('href'))
                      for link in links]

        logger.info(f"Found {len(collections)} collections\n")
        if not collections:
            self._record_failure(
                reason="no_collections",
                url=RabindraScraperConfig.POETRY_INDEX_URL,
            )
        return collections

    def save_collection(self, collection_name: str, poems: List[Dict]):
        """Save poems to JSON"""
        safe_name = collection_name.replace('/', '_').replace(' ', '_')
        filepath = self.output_dir / f"{safe_name}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(poems, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✓ Saved {len(poems)} poems")

    def save_summary(self, collections_info: List[Dict]):
        """Save summary"""
        summary = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "collections": collections_info,
            "source": RabindraScraperConfig.BASE_URL
        }

        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def scrape_all(self, specific_collection: Optional[str] = None,
                   limit_per_collection: Optional[int] = None):
        """Scrape all or specific collections"""
        collections = self.get_all_collections()
        if not collections:
            logger.error("No collections found")
            return

        summary = []
        for collection_name, collection_url in collections:
            if specific_collection and specific_collection.lower() not in collection_name.lower():
                continue

            poems = self.scrape_collection(collection_url, collection_name,
                                          limit=limit_per_collection)

            if poems:
                summary.append({
                    "name": collection_name,
                    "poem_count": len(poems),
                    "file": f"{collection_name.replace('/', '_').replace(' ', '_')}.json"
                })
                logger.info(f"✓ {collection_name}: {len(poems)} poems\n")

        self.save_summary(summary)
        if self.failure_count:
            logger.warning(f"Recorded {self.failure_count} failures in {self.failures_path}")
        logger.info(f"✓ Complete! Output: {self.output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape Rabindra Tagore's poetry")
    parser.add_argument('--collection', help='Collection name (partial match OK)')
    parser.add_argument('--output', default=RabindraScraperConfig.OUTPUT_DIR)
    parser.add_argument('--limit', type=int, help='Limit poems per collection (for testing)')
    parser.add_argument('--delay', type=float, default=RabindraScraperConfig.DELAY_BETWEEN_REQUESTS,
                       help='Delay between requests')

    args = parser.parse_args()
    RabindraScraperConfig.DELAY_BETWEEN_REQUESTS = args.delay

    scraper = RabidraScraper(output_dir=args.output)
    scraper.scrape_all(specific_collection=args.collection,
                       limit_per_collection=args.limit)

if __name__ == '__main__':
    main()
