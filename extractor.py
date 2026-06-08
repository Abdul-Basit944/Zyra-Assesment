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
    return _DEADLINE_LABEL_MAP.get(raw.strip().lower())


def fetch_and_clean(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        if r.status_code != 200:
            return "", r.status_code, ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer",
                         "header", "aside", "form", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = re.sub(r"\s{2,}", " ", soup.get_text(separator=" ", strip=True))

        return text[:MAX_TEXT_CHARS], r.status_code, title

    except requests.RequestException:
        return "", 0, ""


def _call_gemini(client: Groq, prompt: str) -> dict:
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
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(30)
            else:
                return {}

    return {}


_OVERVIEW_PROMPT = """Extract university overview data as JSON only.
Return null if unknown.

{
  "university_name": null,
  "city": null,
  "state": null,
  "country": null,
  "postal_code": null,
  "phone": null,
  "email": null
}

Text:
{text}
"""

_ADMISSIONS_PROMPT = """Extract admission deadlines as JSON only.

{
  "deadlines": [
    {
      "deadline_type": null,
      "deadline_date": null,
      "notes": null
    }
  ]
}

Text:
{text}
"""

_TUITION_PROMPT = """Extract tuition data as JSON only.

{
  "tuition_items": [
    {
      "fee_type": null,
      "cost": null,
      "currency": null
    }
  ]
}

Text:
{text}
"""


class UniversityExtractor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def extract(self, admissions_url: str | None = None, tuition_url: str | None = None) -> UniversityData:
        page_metadata = []

        admissions_text = ""
        tuition_text = ""

        if admissions_url:
            text, status, title = fetch_and_clean(admissions_url)
            admissions_text = text
            page_metadata.append(PageMetadata(
                url=admissions_url,
                page_title=title or None,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                status_code=str(status),
            ))

        if tuition_url:
            text, status, title = fetch_and_clean(tuition_url)
            tuition_text = text
            page_metadata.append(PageMetadata(
                url=tuition_url,
                page_title=title or None,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                status_code=str(status),
            ))

        overview_text = admissions_text if len(admissions_text) >= len(tuition_text) else tuition_text

        overview = None
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

        admission_deadlines = []
        if admissions_text:
            raw = _call_gemini(self.client, _ADMISSIONS_PROMPT.format(text=admissions_text))
            for item in raw.get("deadlines", []):
                dtype = _normalise_deadline_type(item.get("deadline_type"))

                admission_deadlines.append(AdmissionDeadline(
                    deadline_type=dtype,
                    deadline_date=item.get("deadline_date"),
                    notes=item.get("notes"),
                ))

        tuition_breakdown = []
        if tuition_text:
            raw = _call_gemini(self.client, _TUITION_PROMPT.format(text=tuition_text))
            for item in raw.get("tuition_items", []):
                cost = None
                try:
                    if item.get("cost") is not None:
                        cost = int(round(float(str(item["cost"]).replace(",", "").replace("$", ""))))
                except Exception:
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
