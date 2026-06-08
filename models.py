from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, EmailStr


class Location(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class Contact(BaseModel):
    phone: Optional[str] = None
    email: Optional[EmailStr] = None


class Overview(BaseModel):
    university_name: Optional[str] = None
    location: Optional[Location] = None
    contact: Optional[Contact] = None


class TuitionItem(BaseModel):
    fee_type: Optional[str] = None
    cost: Optional[int] = None
    currency: Optional[str] = None


class DeadlineType(str, Enum):
    EARLY_DECISION = "Early Decision"
    REGULAR_DECISION = "Regular Decision"
    TRANSFER_ADMISSION = "Transfer Admission"


class AdmissionDeadline(BaseModel):
    deadline_type: Optional[DeadlineType] = None
    deadline_date: Optional[str] = None
    notes: Optional[str] = None


class PageMetadata(BaseModel):
    url: Optional[str] = None
    page_title: Optional[str] = None
    scraped_at: Optional[str] = None
    status_code: Optional[str] = None


class UniversityData(BaseModel):
    overview: Optional[Overview] = None
    tuition_breakdown: List[TuitionItem] = []
    admission_deadlines: List[AdmissionDeadline] = []
    page_metadata: List[PageMetadata] = []
