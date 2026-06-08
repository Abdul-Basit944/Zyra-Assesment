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
    "User-Agent": "Mozilla/5.0 (compatible; UniversityETL/1.0)"
}

POSITIVE_KEYWORDS = [
    "admission", "apply", "enroll", "tuition", "cost", "fees",
    "financial", "financial-aid", "scholarship", "deadline",
    "cost-of-attendance", "undergraduate",
]

NEGATIVE_KEYWORDS = [
    "news", "events", "athletics", "sports", "calendar", "blog",
    "campus-tour", "gallery", "store", "shop", "alumni", "giving",
    "faculty", "staff", "directory", "login", "portal",
    "research", "library",
]

BINARY_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".jpg", ".jpeg",
    ".png", ".gif", ".svg", ".mp4", ".mp3",
    ".css", ".js",
)


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = (parsed.path or "/").rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def is_internal(url: str, seed_url: str) -> bool:
    url_host = urlparse(url).netloc.lower().lstrip("www.")
    seed_host = urlparse(seed_url).netloc.lower().lstrip("www.")
    return url_host == seed_host or url_host.endswith("." + seed_host)


def is_binary(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in BINARY_EXTENSIONS)


def score_url(url: str) -> int:
    lower = url.lower()
    s = 0

    for kw in POSITIVE_KEYWORDS:
        if kw in lower:
            s += 5 if kw in {"admission", "tuition", "cost-of-attendance", "deadline"} else 3

    for kw in NEGATIVE_KEYWORDS:
        if kw in lower:
            s -= 6

    return s


def fetch_page(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        if r.status_code != 200:
            return None, None

        if any(x in r.url.lower() for x in ("login", "signin", "sso", "auth", "portal")):
            return None, None

        if "text/html" not in r.headers.get("Content-Type", ""):
            return None, None

        return r, BeautifulSoup(r.text, "html.parser")

    except requests.RequestException:
        return None, None


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue

        full = urljoin(base_url, href).split("#")[0]

        if full.startswith("http"):
            links.append(full)

    return links


def crawl(seed_url: str) -> dict[str, int]:
    seed_url = normalize_url(seed_url)
    queue = deque([(seed_url, 0)])
    visited = set()
    scores = {}

    while queue and len(visited) < MAX_PAGES:
        url, depth = queue.popleft()
        url = normalize_url(url)

        if url in visited or depth > MAX_DEPTH or is_binary(url):
            continue

        visited.add(url)
        scores[url] = score_url(url)

        if depth == MAX_DEPTH:
            continue

        time.sleep(CRAWL_DELAY)

        _, soup = fetch_page(url)
        if not soup:
            continue

        for link in extract_links(soup, url):
            link = normalize_url(link)

            if link in visited or not is_internal(link, seed_url) or is_binary(link):
                continue

            link_score = score_url(link)
            scores.setdefault(link, link_score)

            threshold = 0 if depth == 0 else 3
            if link_score >= threshold:
                queue.append((link, depth + 1))

    return scores


def get_top_pages(url_scores: dict[str, int], top_n: int = 10) -> list[str]:
    sorted_pages = sorted(url_scores.items(), key=lambda x: x[1], reverse=True)
    return [url for url, score in sorted_pages if score >= 0][:top_n]
