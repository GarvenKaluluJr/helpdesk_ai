from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ...db import get_db
from ...models.ticket import Ticket
from ...schemas.ticket import TicketCreate, TicketRead

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post(
    "/",
    response_model=TicketRead,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    """
    JSON ticket creation endpoint (API clients).
    """
    ticket = Ticket(
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        body=payload.body,
        status="new",
    )

    if payload.category_hint:
        ticket.category_pred = payload.category_hint

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return ticket
