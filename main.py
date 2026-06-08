"""
main.py
-------
Entry point for the university ETL pipeline.

Flow:
  1. Crawl the seed domain (BFS, max depth 2)
  2. Score and categorise discovered URLs into "admissions" vs "tuition"
  3. Pass the best URL of each category to UniversityExtractor
  4. Validate output through Pydantic (UniversityData)
  5. Write one JSON file per university to ./output/
"""

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import sys

from crawler import crawl
from extractor import UniversityExtractor
from models import UniversityData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Category URL signals ────────────────────────────────────────────────
# Matched against URL path only — no extra fetching needed.

ADMISSIONS_URL_SIGNALS = [
    ("dates-deadlines",       10),
    ("admissions-dates",      10),
    ("admission-deadlines",   10),
    ("apply-deadlines",       10),
    ("deadlines",              8),
    ("admissions-aid/admissions", 6),
    ("admissions-aid/apply",   6),
    ("admissions/apply",       6),
    ("admission",              4),
    ("apply",                  3),
    ("enroll",                 3),
]

TUITION_URL_SIGNALS = [
    ("cost-of-attendance",    10),
    ("tuition-fees",          10),
    ("tuition-and-fees",      10),
    ("tuition-fee",            8),
    ("cost-attend",            8),
    ("tuition",                6),
    ("cost",                   4),
    ("fees",                   3),
    ("financial-aid",          3),
]

# Disqualify pages that superficially match but are not the right page
DISQUALIFY_PATTERNS = [
    "scholarship-programs",
    "scholarship",
    "requirements",
    "virtual-welcome",
    "plan-visit",
    "find-your-counselor",
    "ive-been-admitted",
    "register",
]


def _url_category_score(url: str) -> tuple[int, int]:
    """Return (admissions_score, tuition_score) based on URL path only."""
    lower = url.lower()
    if any(p in lower for p in DISQUALIFY_PATTERNS):
        return 0, 0
    adm = sum(w for p, w in ADMISSIONS_URL_SIGNALS if p in lower)
    tui = sum(w for p, w in TUITION_URL_SIGNALS if p in lower)
    return adm, tui


def select_pages(
    url_scores: dict[str, int],
    seed_url: str,
) -> tuple[str | None, str | None]:
    """
    Select the best admissions and tuition URLs from crawl results
    using URL-path pattern matching only — no extra HTTP fetches.

    Picks the highest-scoring URL in each category. A URL only wins
    the category it scores highest on.
    """
    adm_candidates: list[tuple[int, str]] = []
    tui_candidates: list[tuple[int, str]] = []

    for url in url_scores:
        adm, tui = _url_category_score(url)
        if adm == 0 and tui == 0:
            continue
        if adm >= tui:
            adm_candidates.append((adm, url))
        else:
            tui_candidates.append((tui, url))

    admissions_url = max(adm_candidates, key=lambda x: x[0])[1] if adm_candidates else None
    tuition_url    = max(tui_candidates, key=lambda x: x[0])[1] if tui_candidates else None

    if admissions_url:
        logger.info(f"  → Admissions page: {admissions_url}")
    if tuition_url:
        logger.info(f"  → Tuition page:    {tuition_url}")

    return admissions_url, tuition_url


def run(seed_url: str, name: str, extractor: UniversityExtractor) -> dict:
    """
    Full pipeline for one university.
    Returns a JSON-serialisable dict matching UniversityData schema.
    """
    logger.info(f"── {name} ({seed_url})")

    # Step 1: crawl
    url_scores = crawl(seed_url)
    logger.info(f"  Crawled {len(url_scores)} pages")

    # Step 2: select best admissions + tuition pages
    admissions_url, tuition_url = select_pages(url_scores, seed_url)

    if not admissions_url:
        logger.warning(f"  Could not identify an admissions page for {name}")
    if not tuition_url:
        logger.warning(f"  Could not identify a tuition page for {name}")

    # Step 3: LLM extraction → validated UniversityData
    data: UniversityData = extractor.extract(
        admissions_url=admissions_url,
        tuition_url=tuition_url,
    )

    # Step 4: Pydantic → dict (exclude unset Nones for clean JSON output)
    return data.model_dump(mode="json")


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    universities: dict[str, str] = {
        "bucknell":   "https://www.bucknell.edu/",
        "udc":        "https://www.udc.edu/",
        "salisbury":  "https://www.salisbury.edu/",
    }

    os.makedirs("output", exist_ok=True)
    extractor = UniversityExtractor(api_key=api_key)

    for name, url in universities.items():
        try:
            result = run(url, name, extractor)
            out_path = f"output/{name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved → {out_path}")
        except Exception as e:
            logger.error(f"  Pipeline failed for {name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()