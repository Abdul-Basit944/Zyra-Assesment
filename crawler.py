import requests
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urljoin, urlparse
import time
import logging

logger = logging.getLogger(__name__)

MAX_DEPTH = 2
MAX_PAGES = 60
REQUEST_TIMEOUT = 10
CRAWL_DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; UniversityETL/1.0; +https://github.com/your-repo)"
}

POSITIVE_KEYWORDS = [
    "admission",
    "apply",
    "enroll",
    "tuition",
    "cost",
    "fees",
    "financial",
    "financial-aid",
    "scholarship",
    "deadline",
    "cost-of-attendance",
    "undergraduate",
]

NEGATIVE_KEYWORDS = [
    "news",
    "events",
    "athletics",
    "sports",
    "calendar",
    "blog",
    "campus-tour",
    "gallery",
    "store",
    "shop",
    "alumni",
    "giving",
    "faculty",
    "staff",
    "directory",
    "login",
    "portal",
    "research",
    "library",
]

BINARY_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".jpg", ".jpeg",
    ".png", ".gif", ".svg", ".mp4", ".mp3",
    ".css", ".js",
)


def normalize_url(url: str) -> str:
    """Lowercase netloc, strip fragment and trailing slash."""
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    
    path = path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def is_internal(url: str, seed_domain: str) -> bool:
    """True if url is on the same domain (or a subdomain) as seed_domain."""
    url_host = urlparse(url).netloc.lower().lstrip("www.")
    seed_host = urlparse(seed_domain).netloc.lower().lstrip("www.")
    return url_host == seed_host or url_host.endswith("." + seed_host)


def is_binary(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in BINARY_EXTENSIONS)


def score_url(url: str) -> int:
    """
    Score a URL by keyword signals.
    Higher = more likely to be an admissions/tuition page.
    Negative = almost certainly irrelevant.
    """
    lower = url.lower()
    s = 0

    for kw in POSITIVE_KEYWORDS:
        if kw in lower:
            s += 5 if kw in {"admission", "tuition", "cost-of-attendance", "deadline"} else 3

    for kw in NEGATIVE_KEYWORDS:
        if kw in lower:
            s -= 6

    return s


def fetch_page(url: str) -> tuple[requests.Response | None, BeautifulSoup | None]:
    """
    Fetch a page with error handling.
    Returns (response, soup) or (None, None) on failure.
    Skips pages that require authentication (redirects to login).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        final_url = response.url.lower()
        if any(kw in final_url for kw in ("login", "signin", "sso", "auth", "portal")):
            logger.debug(f"Skipping auth-redirect: {url} → {response.url}")
            return None, None

        if response.status_code != 200:
            logger.debug(f"Non-200 status {response.status_code} for {url}")
            return None, None

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            logger.debug(f"Skipping non-HTML content ({content_type}): {url}")
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")
        return response, soup

    except requests.RequestException as e:
        logger.debug(f"Request failed for {url}: {e}")
        return None, None


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Extract and resolve all href links from a parsed page."""
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        full = urljoin(base_url, href)
        full = full.split("#")[0]
        if full.startswith("http"):
            links.append(full)
    return links


def crawl(seed_url: str) -> dict[str, int]:
    """
    BFS crawl starting from seed_url up to MAX_DEPTH.
    Returns a dict of {normalized_url: score} for all visited pages.

    Design decisions:
    - Depth 0 = seed page (always visited regardless of score)
    - Depth 1 = links from seed; filtered by score >= 0
    - Depth 2 = links from depth-1 pages; filtered by score > 0 (stricter)
    - Binary files and auth redirects are skipped
    - Polite delay between requests
    """
    seed_url = normalize_url(seed_url)
    queue = deque([(seed_url, 0)])
    visited: set[str] = set()
    url_scores: dict[str, int] = {}

    while queue and len(visited) < MAX_PAGES:
        url, depth = queue.popleft()
        url = normalize_url(url)

        if url in visited:
            continue
        if depth > MAX_DEPTH:
            continue
        if is_binary(url):
            continue

        visited.add(url)
        url_scores[url] = score_url(url)

        logger.info(f"[depth={depth}] Crawling: {url} (score={url_scores[url]})")

        if depth == MAX_DEPTH:
            continue

        # Polite delay
        time.sleep(CRAWL_DELAY)

        response, soup = fetch_page(url)
        if soup is None:
            continue

        for link in extract_links(soup, url):
            normalized = normalize_url(link)

            if normalized in visited:
                continue
            if not is_internal(normalized, seed_url):
                continue
            if is_binary(normalized):
                continue

            link_score = score_url(normalized)
            url_scores.setdefault(normalized, link_score)

            threshold = 0 if depth == 0 else 3
            if link_score >= threshold:
                queue.append((normalized, depth + 1))

    logger.info(f"Crawl complete. Visited {len(visited)} pages.")
    return url_scores


def get_top_pages(url_scores: dict[str, int], top_n: int = 10) -> list[str]:
    """
    Return the top N URLs by score, for passing to the extractor.
    Filters out zero-or-negative scored pages (the seed page itself
    may have score 0, so keep anything that was visited).
    """
    sorted_urls = sorted(url_scores.items(), key=lambda x: x[1], reverse=True)
    return [url for url, s in sorted_urls if s >= 0][:top_n]
