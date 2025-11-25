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
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from .api.v1.tickets import router as tickets_router
from .auth import (
    create_access_token,
    get_current_user,
    require_admin,
    verify_password,
    get_password_hash,
)
from .db import get_db, SessionLocal
from .models.ticket import Ticket
from .models.user import User
from .schemas.ticket import TicketCreate

import csv
from io import StringIO


app = FastAPI(title="AI Helpdesk Ticket Classifier")


# -------- Phase 3.1: seed initial users (admin, agent) --------

@app.on_event("startup")
def seed_initial_users():
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(username="admin").first():
            admin = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                role="admin",
            )
            db.add(admin)
        if not db.query(User).filter_by(username="agent").first():
            agent = User(
                username="agent",
                password_hash=get_password_hash("agent123"),
                role="agent",
            )
            db.add(agent)
        db.commit()
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}


# -------- Phase 3.2: login + session (JWT cookie) --------

@app.get("/login", response_class=HTMLResponse)
async def login_form():
    return """
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8" /><title>Login</title></head>
      <body>
        <h1>Helpdesk login</h1>
        <form method="post" action="/login">
          <label>Username:</label><br />
          <input type="text" name="username" required /><br /><br />
          <label>Password:</label><br />
          <input type="password" name="password" required /><br /><br />
          <button type="submit">Login</button>
        </form>
      </body>
    </html>
    """


@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return HTMLResponse(
            content="<h2>Invalid username or password</h2><a href='/login'>Try again</a>",
            status_code=400,
        )

    token = create_access_token({"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("access_token")
    return response


# -------- Phase 3.3: dashboard (needs logged-in user) --------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tickets = db.query(Ticket).order_by(Ticket.created_at.desc()).all()

    rows = ""
    for t in tickets:
        rows += (
            f"<tr><td>{t.id}</td><td>{t.subject}</td>"
            f"<td>{t.status}</td><td>{t.email}</td></tr>"
        )

    return f"""
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8" /><title>Dashboard</title></head>
      <body>
        <h1>Dashboard ({current_user.username}, role={current_user.role})</h1>
        <p><a href="/logout">Logout</a></p>
        <p><a href="/">Create ticket</a></p>
        <table border="1" cellpadding="4" cellspacing="0">
          <thead>
            <tr><th>ID</th><th>Subject</th><th>Status</th><th>Email</th></tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </body>
    </html>
    """


# -------- Phase 2.2/3.3: ticket form (requires login) --------

@app.get("/", response_class=HTMLResponse)
async def ticket_form(current_user: User = Depends(get_current_user)):
    """
    Ticket creation form – only for authenticated users.
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


# -------- Phase 2.1/3.3: HTML /tickets (requires login) --------

@app.post("/tickets", response_class=HTMLResponse)
async def submit_ticket(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    category_hint: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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
        <a href="/">Create another ticket</a><br/>
        <a href="/dashboard">Back to dashboard</a>
      </body>
    </html>
    """


# -------- Phase 2.4/3.3/3.4: admin import (admin-only) --------

@app.post("/admin/import-tickets")
async def import_tickets(
    file: Optional[UploadFile] = File(None),
    tickets: Optional[List[TicketCreate]] = Body(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Admin-only CSV/JSON ticket import.
    """
    imported = 0

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

    if file is not None:
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
        for p in tickets:
            create_from_payload(p)
        db.commit()
        return JSONResponse({"imported": imported, "source": "json"})

    raise HTTPException(
        status_code=400,
        detail="Provide either a CSV file ('file') or a JSON list of tickets in the body.",
    )


# -------- Existing JSON API v1 --------

app.include_router(tickets_router, prefix="/api/v1")
