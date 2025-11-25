# helpdesk_ai/backend/app/api/v1/tickets.py

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ...db import get_db
from ...models.ticket import Ticket
from ...schemas.ticket import TicketCreate, TicketRead
from ...ml.predictor import predict_category


router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post(
    "/",
    response_model=TicketRead,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    """
    JSON ticket creation endpoint (API clients) with ML classification.
    """
    # Run classifier on subject+body
    pred_category, confidence = predict_category(payload.subject, payload.body)

    ticket = Ticket(
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        body=payload.body,
        status="new",
    )

    if pred_category:
        ticket.category_pred = pred_category
        ticket.category_final = pred_category
        ticket.confidence = confidence
    elif payload.category_hint:
        # fallback if no model but client provides a hint
        ticket.category_pred = payload.category_hint
        ticket.category_final = payload.category_hint

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return ticket
