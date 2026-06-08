import json
import logging
import re
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from groq import Groq

from models import (
    AdmissionDeadline,
    Contact,
    DeadlineType,
    Location,
    Overview,
    PageMetadata,
    TuitionItem,
    UniversityData,
)

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UniversityETL/1.0)"}
REQUEST_TIMEOUT = 10
MAX_TEXT_CHARS = 6000  

_DEADLINE_LABEL_MAP = {
    "early decision": DeadlineType.EARLY_DECISION,
    "early decision i": DeadlineType.EARLY_DECISION,
    "early decision ii": DeadlineType.EARLY_DECISION,
    "ed": DeadlineType.EARLY_DECISION,
    "ed i": DeadlineType.EARLY_DECISION,
    "ed ii": DeadlineType.EARLY_DECISION,
    "early action": DeadlineType.EARLY_DECISION,     
    "restrictive early action": DeadlineType.EARLY_DECISION,
    "regular decision": DeadlineType.REGULAR_DECISION,
    "regular admission": DeadlineType.REGULAR_DECISION,
    "rd": DeadlineType.REGULAR_DECISION,
    "rolling admission": DeadlineType.REGULAR_DECISION,
    "transfer": DeadlineType.TRANSFER_ADMISSION,
    "transfer admission": DeadlineType.TRANSFER_ADMISSION,
    "transfer application": DeadlineType.TRANSFER_ADMISSION,
    "transfer student": DeadlineType.TRANSFER_ADMISSION,
}


def _normalise_deadline_type(raw: str | None) -> DeadlineType | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _DEADLINE_LABEL_MAP.get(key)

def fetch_and_clean(url: str) -> tuple[str, int, str]:
    """
    Fetch a URL and return (clean_text, status_code, page_title).
    Strips nav/footer/scripts, returns plain text for LLM consumption.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        status = resp.status_code

        if status != 200:
            return "", status, ""

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer",
                         "header", "aside", "form", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)

        text = re.sub(r"\s{2,}", " ", text)

        return text[:MAX_TEXT_CHARS], status, title

    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return "", 0, ""

def _call_gemini(client: Groq, prompt: str) -> dict:
    """
    Call Groq and parse the JSON response.
    Retries up to 3 times on rate limit errors.
    Returns empty dict on unrecoverable failure.
    """
    import time

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 30
                logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/3)")
                time.sleep(wait)
            else:
                logger.warning(f"Groq error: {e}")
                return {}

    logger.warning("Groq failed after 3 attempts, returning empty result")
    return {}



_OVERVIEW_PROMPT = """
You are extracting structured data from a university website page.

Extract the following fields from the page text below and return ONLY a valid JSON object.
Return null for any field you cannot find with reasonable confidence. Do NOT fabricate values.

Fields to extract:
{{
  "university_name": string or null,
  "city": string or null,
  "state": string or null,
  "country": string or null (default "United States" if clearly a US university),
  "postal_code": string or null,
  "phone": string or null,
  "email": string or null
}}

Rules:
- phone: include country code if present, normalize format, e.g. "+1 (570) 577-2000"
- email: must be a valid email address; return null if unsure
- university_name: official full name, not abbreviation

Page text:
{text}
"""

_ADMISSIONS_PROMPT = """
You are extracting structured admissions deadline data from a university website.

From the page text below, extract all admission deadlines and return ONLY a valid JSON object.
Return null for any field you cannot determine with reasonable confidence. Do NOT fabricate dates.

Return format:
{{
  "deadlines": [
    {{
      "deadline_type": one of exactly ["Early Decision", "Regular Decision", "Transfer Admission"] or null,
      "deadline_date": string date e.g. "November 1, 2024" or null,
      "notes": any extra context e.g. "for Fall 2025 entry" or null
    }}
  ]
}}

Rules for deadline_type:
- "Early Decision" covers: Early Decision I/II, Early Action, Restrictive Early Action
- "Regular Decision" covers: Regular Decision, Rolling Admission, Regular Admission
- "Transfer Admission" covers: any deadline specifically for transfer students
- If you cannot map the label to one of the three values above, return null for that field

Page text:
{text}
"""

_TUITION_PROMPT = """
You are extracting structured tuition and cost data from a university website.

From the page text below, extract all tuition and fee line items and return ONLY a valid JSON object.
Return null for any field you cannot determine with reasonable confidence. Do NOT fabricate amounts.

Return format:
{{
  "tuition_items": [
    {{
      "fee_type": string describing the fee e.g. "Tuition", "Room & Board", "Student Fees", "Books & Supplies",
      "cost": integer dollar amount (strip $ and commas, round to nearest integer) or null,
      "currency": "USD" or null
    }}
  ]
}}

Rules:
- cost must be an integer (not a string, not a float)
- Include per-year amounts where available; note in fee_type if per-semester
- If a range is given (e.g. "$1,200–$1,800"), use the lower bound
- Only extract concrete dollar amounts; skip percentage-based fees
- currency should be "USD" for all US universities unless stated otherwise

Page text:
{text}
"""


class UniversityExtractor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def extract(
        self,
        admissions_url: str | None = None,
        tuition_url: str | None = None,
    ) -> UniversityData:
        """
        Given up to two URLs (admissions page + tuition page),
        extract and return a validated UniversityData object.
        """
        page_metadata: list[PageMetadata] = []

        admissions_text, admissions_title = "", ""
        tuition_text, tuition_title = "", ""

        if admissions_url:
            text, status, title = fetch_and_clean(admissions_url)
            admissions_text, admissions_title = text, title
            page_metadata.append(PageMetadata(
                url=admissions_url,
                page_title=title or None,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                status_code=str(status),
            ))

        if tuition_url:
            text, status, title = fetch_and_clean(tuition_url)
            tuition_text, tuition_title = text, title
            page_metadata.append(PageMetadata(
                url=tuition_url,
                page_title=title or None,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                status_code=str(status),
            ))

        overview_text = admissions_text if len(admissions_text) >= len(tuition_text) else tuition_text

        overview: Overview | None = None
        if overview_text:
            raw = _call_gemini(self.client, _OVERVIEW_PROMPT.format(text=overview_text))
            if raw:
                overview = Overview(
                    university_name=raw.get("university_name"),
                    location=Location(
                        city=raw.get("city"),
                        state=raw.get("state"),
                        country=raw.get("country"),
                        postal_code=raw.get("postal_code"),
                    ) if any(raw.get(k) for k in ("city", "state", "country", "postal_code")) else None,
                    contact=Contact(
                        phone=raw.get("phone"),
                        email=raw.get("email"),
                    ) if any(raw.get(k) for k in ("phone", "email")) else None,
                )
              
        admission_deadlines: list[AdmissionDeadline] = []
        if admissions_text:
            raw = _call_gemini(self.client, _ADMISSIONS_PROMPT.format(text=admissions_text))
            for item in raw.get("deadlines", []):
                dtype = _normalise_deadline_type(item.get("deadline_type"))
                try:
                    if dtype is None and item.get("deadline_type"):
                        dtype = DeadlineType(item["deadline_type"])
                except ValueError:
                    dtype = None

                admission_deadlines.append(AdmissionDeadline(
                    deadline_type=dtype,
                    deadline_date=item.get("deadline_date"),
                    notes=item.get("notes"),
                ))

        tuition_breakdown: list[TuitionItem] = []
        if tuition_text:
            raw = _call_gemini(self.client, _TUITION_PROMPT.format(text=tuition_text))
            for item in raw.get("tuition_items", []):
                raw_cost = item.get("cost")
                cost: int | None = None
                if raw_cost is not None:
                    try:
                        cost = int(round(float(str(raw_cost).replace(",", "").replace("$", ""))))
                    except (ValueError, TypeError):
                        cost = None

                tuition_breakdown.append(TuitionItem(
                    fee_type=item.get("fee_type"),
                    cost=cost,
                    currency=item.get("currency", "USD"),
                ))

        return UniversityData(
            overview=overview,
            tuition_breakdown=tuition_breakdown,
            admission_deadlines=admission_deadlines,
            page_metadata=page_metadata,
        )
