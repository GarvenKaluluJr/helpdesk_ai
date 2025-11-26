# helpdesk_ai/backend/app/api/v1/tickets.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...db import get_db
from ...models.ticket import Ticket
from ...schemas.ticket import TicketCreate, TicketRead
from ...ml.predictor import predict_category, compute_priority, route_to_queue

router = APIRouter(prefix="/tickets", tags=["tickets"])

@router.post(
    "/",
    response_model=TicketRead,
    status_code=status.HTTP_201_CREATED,
)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    text = f"{payload.subject or ''} {payload.body or ''}".strip()

    category_pred, confidence = (None, None)
    if text:
        category_pred, confidence = predict_category(text)

    priority_pred = compute_priority(text, category_pred)
    queue_value = route_to_queue(category_pred)

    ticket = Ticket(
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        body=payload.body,
        status="new",
        category_pred=category_pred,
        category_final=category_pred,
        confidence=confidence,
        priority_pred=priority_pred,
        priority_final=priority_pred,
        queue=queue_value,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket

@router.get("/{ticket_id}", response_model=TicketRead)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket
