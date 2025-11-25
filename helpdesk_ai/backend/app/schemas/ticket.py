# backend/app/schemas/ticket.py

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class TicketCreate(BaseModel):
    name: str
    email: EmailStr
    subject: str
    body: str


class TicketRead(BaseModel):
    id: int
    name: str
    email: EmailStr
    subject: str
    body: str

    category_pred: Optional[str] = None
    category_final: Optional[str] = None
    priority_pred: Optional[str] = None
    priority_final: Optional[str] = None
    queue: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    confidence: Optional[float] = None

    class Config:
        orm_mode = True
