# helpdesk_ai/backend/app/main.py

from typing import List, Optional

from fastapi import (
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    FastAPI,
)
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from .api.v1.tickets import router as tickets_router
from .db import get_db
from .models.ticket import Ticket
from .schemas.ticket import TicketCreate

import csv
from io import StringIO


app = FastAPI(title="AI Helpdesk Ticket Classifier")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# -------------------------------
# Phase 2.2 – Simple HTML form
# -------------------------------

@app.get("/", response_class=HTMLResponse)
async def ticket_form():
    """
    Simple HTML form that posts to /tickets .
    """
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Create Ticket</title>
      </head>
      <body>
        <h1>Create Support Ticket</h1>
        <form method="post" action="/tickets">
          <label>Name:</label><br />
          <input type="text" name="name" required /><br /><br />

          <label>Email:</label><br />
          <input type="email" name="email" required /><br /><br />

          <label>Subject:</label><br />
          <input type="text" name="subject" required /><br /><br />

          <label>Message:</label><br />
          <textarea name="body" rows="5" cols="40" required></textarea><br /><br />

          <label>Category hint (optional):</label><br />
          <input type="text" name="category_hint" placeholder="Technical, Finance, etc." /><br /><br />

          <button type="submit">Submit ticket</button>
        </form>
      </body>
    </html>
    """


# -------------------------------
# Phase 2.1 & 2.3 – /tickets (HTML submission)
# -------------------------------

@app.post("/tickets", response_class=HTMLResponse)
async def submit_ticket(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    category_hint: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    HTML form submission endpoint .

    - Validates name, email, subject, body (via form required fields).
    - Optional category_hint.
    - Sets status = "new" and created_at via DB defaults.
    """
    ticket = Ticket(
        name=name,
        email=email,
        subject=subject,
        body=body,
        status="new",
    )

    if category_hint:
        ticket.category_pred = category_hint

    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return f"""
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8" /><title>Ticket created</title></head>
      <body>
        <h2>Ticket created successfully</h2>
        <p><strong>ID:</strong> {ticket.id}</p>
        <p><strong>Subject:</strong> {ticket.subject}</p>
        <p><strong>Status:</strong> {ticket.status}</p>
        <a href="/">Create another ticket</a>
      </body>
    </html>
    """


# -----------------------------------------------
# Phase 2.4 – /admin/import-tickets (CSV or JSON)
# -----------------------------------------------

@app.post("/admin/import-tickets")
async def import_tickets(
    file: Optional[UploadFile] = File(None),
    tickets: Optional[List[TicketCreate]] = Body(None),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint to bulk-import tickets.

    Supports:
      - CSV file uploaded as "file" (columns: name,email,subject,body,category_hint)
      - JSON body: list[TicketCreate]
    """
    imported = 0

    # Helper to reuse creation logic for each payload
    def create_from_payload(p: TicketCreate):
        nonlocal imported
        ticket = Ticket(
            name=p.name,
            email=p.email,
            subject=p.subject,
            body=p.body,
            status="new",
        )
        if p.category_hint:
            ticket.category_pred = p.category_hint
        db.add(ticket)
        imported += 1
        return ticket

    if file is not None:
        # CSV import
        raw = await file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

        reader = csv.DictReader(StringIO(text))
        for row in reader:
            payload = TicketCreate(
                name=row["name"],
                email=row["email"],
                subject=row["subject"],
                body=row["body"],
                category_hint=row.get("category_hint"),
            )
            create_from_payload(payload)

        db.commit()
        return JSONResponse({"imported": imported, "source": "csv"})

    if tickets is not None:
        # JSON import: tickets is list[TicketCreate]
        for p in tickets:
            create_from_payload(p)
        db.commit()
        return JSONResponse({"imported": imported, "source": "json"})

    raise HTTPException(
        status_code=400,
        detail="Provide either a CSV file ('file') or a JSON list of tickets in the body.",
    )


# -------------------------------
# Existing JSON API v1
# -------------------------------

app.include_router(tickets_router, prefix="/api/v1")
