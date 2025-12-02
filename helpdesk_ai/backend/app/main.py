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
from sqlalchemy import and_, or_
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
from .models.ticket_history import TicketHistory
from .models.user import User
from .schemas.ticket import TicketCreate, TicketRead, TicketUpdate

import csv
import json   
from io import StringIO

from .models.training_sample import TrainingSample   
from .models.training_run import TrainingRun         
from .ml.train_classifier import train_and_save_from_db  
from .ml.predictor import predict_category, compute_priority, route_to_queue, PREDICTOR_LOADED

from pydantic import EmailStr
app = FastAPI(title="AI Helpdesk Ticket Classifier")

# Allowed values & validators
# These are the canonical labels used everywhere in the project
ALLOWED_CATEGORIES = ["Account", "Administration", "Financy", "General", "Technical"]
ALLOWED_PRIORITIES = ["Low", "Medium", "High"]
ALLOWED_QUEUES = ["IT", "Finance", "Admissions", "General"]

_CATEGORY_CANON = {c.lower(): c for c in ALLOWED_CATEGORIES}
_PRIORITY_CANON = {p.lower(): p for p in ALLOWED_PRIORITIES}
_QUEUE_CANON = {q.lower(): q for q in ALLOWED_QUEUES}


def validate_category_final(value: Optional[str]) -> Optional[str]:
    """
    Normalize + validate category_final coming from UI / API.
    Empty string -> None. Case-insensitive.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key not in _CATEGORY_CANON:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid category_final '{raw}'. "
                f"Allowed: {', '.join(ALLOWED_CATEGORIES)}"
            ),
        )
    return _CATEGORY_CANON[key]


def validate_priority_final(value: Optional[str]) -> Optional[str]:
    """
    Normalize + validate priority_final. Case-insensitive.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key not in _PRIORITY_CANON:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid priority_final '{raw}'. "
                f"Allowed: {', '.join(ALLOWED_PRIORITIES)}"
            ),
        )
    return _PRIORITY_CANON[key]


def validate_queue(value: Optional[str]) -> Optional[str]:
    """
    Normalize + validate queue name. Case-insensitive.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key not in _QUEUE_CANON:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid queue '{raw}'. "
                f"Allowed: {', '.join(ALLOWED_QUEUES)}"
            ),
        )
    return _QUEUE_CANON[key]


#Seed admin

@app.on_event("startup")
def seed_initial_users():
    db = SessionLocal()
    try:
        # only admin
        if not db.query(User).filter_by(username="admin").first():
            admin = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                role="admin",
            )
            db.add(admin)
        db.commit()
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {"status": "ok"}



# Login / Logout

@app.get("/login", response_class=HTMLResponse)
async def login_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <title>Helpdesk Login</title>
        <style>
          * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, sans-serif;
          }

          body {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at top, #1f2937, #020617);
            color: #e5e7eb;
          }

          .login-wrapper {
            width: 100%;
            max-width: 420px;
            padding: 32px 24px;
          }

          .card {
            background: rgba(15, 23, 42, 0.9);
            border-radius: 18px;
            padding: 32px 28px 28px;
            box-shadow:
              0 18px 45px rgba(0, 0, 0, 0.55),
              0 0 0 1px rgba(148, 163, 184, 0.15);
            backdrop-filter: blur(16px);
          }

          .card-header {
            text-align: center;
            margin-bottom: 24px;
          }

          .card-title {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }

          .card-subtitle {
            margin-top: 8px;
            font-size: 0.9rem;
            color: #9ca3af;
          }

          .field {
            margin-bottom: 18px;
          }

          .field label {
            display: block;
            font-size: 0.85rem;
            margin-bottom: 6px;
            color: #d1d5db;
          }

          .field input {
            width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            padding: 10px 14px;
            background: rgba(15, 23, 42, 0.9);
            color: #e5e7eb;
            outline: none;
            font-size: 0.95rem;
            transition: border-color 0.18s ease, box-shadow 0.18s ease,
                        background 0.18s ease;
          }

          .field input:focus {
            border-color: #22c55e;
            box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.35);
            background: rgba(15, 23, 42, 1);
          }

          .actions {
            margin-top: 16px;
          }

          .btn-primary {
            width: 100%;
            border: none;
            border-radius: 999px;
            padding: 10px 16px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            color: #020617;
            background: linear-gradient(90deg, #f97316, #22c55e, #06b6d4);
            background-size: 200% 100%;
            transition: transform 0.12s ease, box-shadow 0.12s ease,
                        background-position 0.35s ease;
            box-shadow: 0 12px 30px rgba(34, 197, 94, 0.3);
          }

          .btn-primary:hover {
            background-position: 100% 0;
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(34, 197, 94, 0.4);
          }

          .meta-row {
            display: flex;
            justify-content: space-between;
            margin-top: 16px;
            font-size: 0.8rem;
            color: #9ca3af;
          }

          .meta-row span {
            opacity: 0.9;
          }

          .brand {
            margin-top: 18px;
            text-align: center;
            font-size: 0.75rem;
            color: #6b7280;
          }

          .brand strong {
            color: #a5b4fc;
          }
        </style>
      </head>
      <body>
        <div class="login-wrapper">
          <div class="card">
            <div class="card-header">
              <div class="card-title">Helpdesk Login</div>
              <p class="card-subtitle">Sign in to manage and triage tickets.</p>
            </div>

            <form method="post" action="/login">
              <div class="field">
                <label for="username">Username</label>
                <input id="username" name="username" type="text" required />
              </div>

              <div class="field">
                <label for="password">Password</label>
                <input id="password" name="password" type="password" required />
              </div>

              <div class="actions">
                <button class="btn-primary" type="submit">Sign in</button>
              </div>

              <div class="meta-row">
                <span>Garven / Kalulu Jr</span>
                <span>AI Helpdesk Console</span>
              </div>
            </form>
          </div>

          <p class="brand">
            Powered by <strong>AI Helpdesk Ticket Classifier</strong>
          </p>
        </div>
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
    response = RedirectResponse(url="/tickets", status_code=302)
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



# Dashboard redirect → /tickets

@app.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/tickets", status_code=302)


# Helpers for display

def format_category_display(ticket: Ticket) -> str:
    if ticket.category_final:
        # Final exists
        base = f"Final: {ticket.category_final}"
        if ticket.category_pred:
            conf_str = (
                f", {ticket.confidence:.2f}" if ticket.confidence is not None else ""
            )
            base += f" (Predicted: {ticket.category_pred}{conf_str})"
        return base
    else:
        # No final, show predicted if exists
        if ticket.category_pred:
            conf_str = (
                f" (confidence {ticket.confidence:.2f})"
                if ticket.confidence is not None
                else ""
            )
            return f"Predicted: {ticket.category_pred}{conf_str}"
        return "—"


def format_priority_display(ticket: Ticket) -> str:
    if ticket.priority_final:
        base = f"Final: {ticket.priority_final}"
        if ticket.priority_pred:
            base += f" (Predicted: {ticket.priority_pred})"
        return base
    else:
        if ticket.priority_pred:
            return f"Predicted: {ticket.priority_pred}"
        return "—"


#Ticket list + filters + pagination

@app.get("/tickets", response_class=HTMLResponse)
async def list_tickets(
    request: Request,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    queue: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Ticket)

    if status:
        query = query.filter(Ticket.status == status)

    if queue:
        query = query.filter(Ticket.queue == queue)

    if priority:
        query = query.filter(Ticket.priority_final == priority)

    if category:
        query = query.filter(
            or_(
                Ticket.category_final == category,
                and_(
                    Ticket.category_final.is_(None),
                    Ticket.category_pred == category,
                ),
            )
        )

    total = query.count()

    page = max(page, 1)
    page_size = max(1, min(page_size, 50))
    tickets = (
        query.order_by(Ticket.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    rows = ""
    for t in tickets:
        cat = format_category_display(t)
        prio = format_priority_display(t)
        queue_str = t.queue or "—"
        created_str = t.created_at.isoformat(sep=" ", timespec="seconds")
        rows += (
            f"<tr>"
            f"<td><a class='id-link' href='/tickets/{t.id}'>{t.id}</a></td>"
            f"<td class='subject-cell'>{t.subject}</td>"
            f"<td>{cat}</td>"
            f"<td>{prio}</td>"
            f"<td>{queue_str}</td>"
            f"<td><span class='status-pill'>{t.status}</span></td>"
            f"<td>{created_str}</td>"
            f"</tr>"
        )

    base_url = "/tickets"
    prev_link = ""
    next_link = ""
    if page > 1:
        prev_link = f"<a href='{base_url}?page={page-1}&page_size={page_size}'>Prev</a>"
    if page * page_size < total:
        next_link = f"<a href='{base_url}?page={page+1}&page_size={page_size}'>Next</a>"

    filters_info = (
        f"category {category or ''}, "
        f"priority {priority or ''}, "
        f"queue {queue or ''}, "
        f"status {status or ''}"
    )
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Ticket Dashboard</title>
      </head>
      <body>
        <h1>Ticket Dashboard ({current_user.username})</h1>
        <style>
          * {{
            box-sizing: border-box;
          }}
          body {{
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000 100%);
            color: #e5e7eb;
          }}
          a {{
            color: #38bdf8;
            text-decoration: none;
          }}
          a:hover {{
            text-decoration: underline;
          }}
          .nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 32px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            backdrop-filter: blur(14px);
          }}
          .nav-title {{
            font-weight: 600;
            letter-spacing: 0.06em;
            font-size: 14px;
            text-transform: uppercase;
            color: #9ca3af;
          }}
          .nav-title span {{
            color: #e5e7eb;
          }}
          .nav-links a {{
            margin-left: 16px;
            font-size: 14px;
          }}
          .nav-links a.primary-link {{
            padding: 6px 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #f97316);
            color: #020617;
            font-weight: 600;
          }}
          .nav-links a.primary-link:hover {{
            filter: brightness(1.05);
            text-decoration: none;
          }}
          .page {{
            max-width: 1200px;
            margin: 32px auto 40px;
            padding: 0 24px 24px;
          }}
          .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 16px;
          }}
          .page-header h1 {{
            margin: 0;
            font-size: 24px;
            letter-spacing: 0.03em;
          }}
          .page-header .meta {{
            font-size: 13px;
            color: #9ca3af;
          }}
          .filters-card {{
            background: radial-gradient(circle at top left, #0f172a, #020617);
            border-radius: 16px;
            padding: 16px 20px 12px;
            border: 1px solid rgba(148, 163, 184, 0.3);
            box-shadow:
              0 18px 45px rgba(15, 23, 42, 0.9),
              0 0 0 1px rgba(15, 23, 42, 0.9);
            margin-bottom: 20px;
          }}
          .filters-current {{
            font-size: 13px;
            color: #9ca3af;
            margin-bottom: 8px;
          }}
          .filters-form {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px 16px;
            align-items: center;
          }}
          .filters-form label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
          }}
          .filters-form input {{
            background: rgba(15, 23, 42, 0.9);
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #e5e7eb;
            padding: 6px 10px;
            font-size: 13px;
            min-width: 160px;
          }}
          .filters-form input:focus {{
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.5);
          }}
          .filters-form button {{
            padding: 7px 16px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(90deg, #22c55e, #f97316);
            color: #020617;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
          }}
          .filters-form button:hover {{
            filter: brightness(1.05);
          }}
          .total-text {{
            font-size: 13px;
            color: #9ca3af;
            margin: 10px 0 6px;
          }}
          .table-wrapper {{
            overflow: auto;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: rgba(15, 23, 42, 0.96);
            box-shadow:
              0 22px 60px rgba(15, 23, 42, 0.95),
              0 0 0 1px rgba(15, 23, 42, 0.9);
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
          }}
          thead {{
            background: radial-gradient(circle at top left, #0f172a, #020617);
          }}
          th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(30, 41, 59, 0.9);
            text-align: left;
            white-space: nowrap;
          }}
          th {{
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
          }}
          tbody tr:hover {{
            background: rgba(15, 23, 42, 0.9);
          }}
          .subject-cell {{
            max-width: 320px;
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
          }}
          .id-link {{
            font-variant-numeric: tabular-nums;
          }}
          .status-pill {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.45);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.09em;
          }}
          .pagination {{
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            font-size: 13px;
            margin-top: 10px;
          }}
        </style>
      </head>
      <body>
        <header class="nav">
          <div class="nav-title">
            AI Helpdesk • <span>{current_user.username}</span>
          </div>
          <div class="nav-links">
            <a href="/tickets">Dashboard</a>
            <a href="/admin/dataset">Training dataset</a>
            <a href="/logout">Logout</a>
          </div>
        </header>

        <main class="page">
          <div class="page-header">
            <h1>Ticket Dashboard</h1>
            <div class="meta">Agent console · real-time classification & routing</div>
          </div>

          <section class="filters-card">
            <div class="filters-current">
              Filters (current: {filters_info})
            </div>
            <form class="filters-form" method="get" action="/tickets">
              <div>
                <label>Category</label><br/>
                <input type="text" name="category" value="{category or ''}" />
              </div>
              <div>
                <label>Priority</label><br/>
                <input type="text" name="priority" value="{priority or ''}" />
              </div>
              <div>
                <label>Queue</label><br/>
                <input type="text" name="queue" value="{queue or ''}" />
              </div>
              <div>
                <label>Status</label><br/>
                <input type="text" name="status" value="{status or ''}" />
              </div>
              <div>
                <label>&nbsp;</label><br/>
                <button type="submit">Apply filters</button>
              </div>
            </form>
            <p class="total-text">Total tickets matching filters: {total}</p>
          </section>

          <section class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Subject</th>
                  <th>Category</th>
                  <th>Priority</th>
                  <th>Queue</th>
                  <th>Status</th>
                  <th>Created At</th>
                </tr>
              </thead>
              <tbody>
                {rows}
              </tbody>
            </table>
          </section>

          <div class="pagination">
            <span>{prev_link}</span>
            <span>{next_link}</span>
          </div>
        </main>
      </body>
    </html>
    """


# Ticket detail (HTML)

@app.get("/tickets/{ticket_id}", response_class=HTMLResponse)
async def ticket_detail(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    cat_display = format_category_display(ticket)
    prio_display = format_priority_display(ticket)

    category_final = ticket.category_final or ""
    priority_final = ticket.priority_final or ""
    queue_value = ticket.queue or ""
    status_value = ticket.status

    confidence_str = (
        f"{ticket.confidence:.2f}" if ticket.confidence is not None else "N/A"
    )

    created_str = ticket.created_at.isoformat(sep=" ", timespec="seconds")

    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Ticket {ticket.id}</title>
        <style>
          * {{
            box-sizing: border-box;
          }}
          body {{
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: radial-gradient(circle at top, #1e293b 0, #020617 45%, #000 100%);
            color: #e5e7eb;
          }}
          a {{
            color: #38bdf8;
            text-decoration: none;
          }}
          a:hover {{
            text-decoration: underline;
          }}
          .nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 32px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            backdrop-filter: blur(14px);
          }}
          .nav-title {{
            font-weight: 600;
            letter-spacing: 0.06em;
            font-size: 14px;
            text-transform: uppercase;
            color: #9ca3af;
          }}
          .nav-title span {{
            color: #e5e7eb;
          }}
          .nav-links a {{
            margin-left: 16px;
            font-size: 14px;
          }}
          .page {{
            max-width: 960px;
            margin: 32px auto 40px;
            padding: 0 24px 24px;
          }}
          .card {{
            background: radial-gradient(circle at top left, #0f172a, #020617);
            border-radius: 20px;
            padding: 20px 22px 22px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow:
              0 22px 60px rgba(15, 23, 42, 0.95),
              0 0 0 1px rgba(15, 23, 42, 0.9);
          }}
          h1 {{
            margin-top: 0;
            margin-bottom: 6px;
            font-size: 22px;
            letter-spacing: 0.05em;
          }}
          .meta {{
            font-size: 13px;
            color: #9ca3af;
            margin-bottom: 16px;
          }}
          .section-title {{
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9ca3af;
            margin-top: 18px;
            margin-bottom: 8px;
          }}
          .field-row {{
            display: flex;
            gap: 16px;
            margin-bottom: 6px;
            font-size: 14px;
          }}
          .field-label {{
            width: 110px;
            color: #9ca3af;
          }}
          pre {{
            background: rgba(15, 23, 42, 0.85);
            border-radius: 12px;
            padding: 10px 12px;
            font-size: 13px;
            max-height: 260px;
            overflow: auto;
          }}
          form {{
            margin-top: 8px;
          }}
          .form-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px 18px;
          }}
          label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
          }}
          input[type="text"] {{
            width: 100%;
            margin-top: 4px;
            background: rgba(15, 23, 42, 0.9);
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            color: #e5e7eb;
            padding: 6px 10px;
            font-size: 13px;
          }}
          input[type="text"]:focus {{
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.5);
          }}
          .actions {{
            margin-top: 18px;
            display: flex;
            justify-content: flex-end;
          }}
          button {{
            padding: 8px 18px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(90deg, #22c55e, #f97316);
            color: #020617;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
          }}
          button:hover {{
            filter: brightness(1.05);
          }}
          .breadcrumbs {{
            font-size: 13px;
            margin-bottom: 10px;
          }}
        </style>
      </head>
      <body>
        <header class="nav">
          <div class="nav-title">
            AI Helpdesk • <span>{current_user.username}</span>
          </div>
        </header>

        <main class="page">
          <div class="breadcrumbs">
            <a href="/tickets">← Back to list</a>
          </div>
          <section class="card">
            <h1>Ticket #{ticket.id}</h1>
            <div class="meta">
              Created at {created_str} · Status: {ticket.status} · Queue: {ticket.queue or "—"}
            </div>

            <div class="section-title">Basic info</div>
            <div class="field-row">
              <div class="field-label">Subject</div>
              <div>{ticket.subject}</div>
            </div>
            <div class="field-row">
              <div class="field-label">From</div>
              <div>{ticket.name} &lt;{ticket.email}&gt;</div>
            </div>

            <div class="section-title">Message body</div>
            <pre>{ticket.body}</pre>

            <div class="section-title">Category &amp; priority</div>
            <div class="field-row">
              <div class="field-label">Category</div>
              <div>{cat_display}</div>
            </div>
            <div class="field-row">
              <div class="field-label">Priority</div>
              <div>{prio_display}</div>
            </div>
            <div class="field-row">
              <div class="field-label">Predicted</div>
              <div>
                {ticket.category_pred or '—'} (confidence {confidence_str}), 
                priority {ticket.priority_pred or '—'}
              </div>
            </div>

            <div class="section-title">Manual edit</div>
            <form method="post" action="/tickets/{ticket.id}/edit">
              <div class="form-grid">
                <div>
                  <label>Final category</label>
                  <input type="text" name="category_final" value="{category_final}" />
                </div>
                <div>
                  <label>Final priority</label>
                  <input type="text" name="priority_final" value="{priority_final}" />
                </div>
                <div>
                  <label>Queue</label>
                  <input type="text" name="queue" value="{queue_value}" />
                </div>
                <div>
                  <label>Status</label>
                  <input type="text" name="status" value="{status_value}" />
                </div>
              </div>
              <div class="actions">
                <button type="submit">Save changes</button>
              </div>
            </form>
          </section>
        </main>
      </body>
    </html>
    """

# Allow opening /tickets/{id}/edit directly in the browser
@app.get("/tickets/{ticket_id}/edit", response_class=HTMLResponse)
async def ticket_edit_get(
    ticket_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Re-use the same detail renderer (shows the form)
    return await ticket_detail(ticket_id=ticket_id, current_user=current_user, db=db)


#ticket_history

@app.post("/tickets/{ticket_id}/edit")
async def ticket_edit_html(
    ticket_id: int,
    category_final: Optional[str] = Form(None),
    priority_final: Optional[str] = Form(None),
    queue: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    changes: dict = {}

    def maybe_update(field: str, raw_value: Optional[str], validator=None):
        if raw_value is None:
            return
        # normalize/validate
        if validator is not None:
            new_val = validator(raw_value)
        else:
            new_val = raw_value.strip() or None
        old_val = getattr(ticket, field)
        if old_val != new_val:
            setattr(ticket, field, new_val)
            changes[field] = (old_val, new_val)

    # Use the validators for consistency
    maybe_update("category_final", category_final, validator=validate_category_final)
    maybe_update("priority_final", priority_final, validator=validate_priority_final)
    maybe_update("queue", queue, validator=validate_queue)
    if status is not None:
        maybe_update("status", status)

    # Log changes into ticket_history
    for field, (old, new) in changes.items():
        history = TicketHistory(
            ticket_id=ticket.id,
            field=field,
            old_value=old if old is not None else "",
            new_value=new if new is not None else "",
            changed_by=current_user.id,
        )
        db.add(history)

    db.commit()

    # After POST, always go back to the detail page (no blank /edit)
    return RedirectResponse(url=f"/tickets/{ticket_id}", status_code=303)


#JSON PATCH /tickets/{id}

@app.patch("/tickets/{ticket_id}", response_model=TicketRead)
async def ticket_edit_api(
    ticket_id: int,
    payload: TicketUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    JSON-level edit for category_final, priority_final, queue, status.
    Also records ticket_history entries.
    """
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    changes = {}
    data = payload.model_dump(exclude_unset=True)

    for field, new_val in data.items():
        # Normalize + validate same as HTML form
        if field == "category_final":
            new_val = validate_category_final(new_val)
        elif field == "priority_final":
            new_val = validate_priority_final(new_val)
        elif field == "queue":
            new_val = validate_queue(new_val)

        old_val = getattr(ticket, field)
        if old_val != new_val:
            setattr(ticket, field, new_val)
            changes[field] = (old_val, new_val)

    for field, (old, new) in changes.items():
        history = TicketHistory(
            ticket_id=ticket.id,
            field=field,
            old_value=old if old is not None else "",
            new_value=new if new is not None else "",
            changed_by=current_user.id,
        )
        db.add(history)

    db.commit()
    db.refresh(ticket)
    return ticket

#  Landing page + public ticket subbmission form 
@app.get("/public/ticket", response_class=HTMLResponse)
async def public_ticket_form():
    """
    Public ticket form for students/guests.
    No login required.
    """
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Create Support Ticket</title>
        <style>
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
          }
          .card {
            background: #020617;
            border-radius: 16px;
            padding: 28px 34px;
            box-shadow: 0 22px 45px rgba(15,23,42,0.75);
            width: 100%;
            max-width: 640px;
            border: 1px solid rgba(148,163,184,0.2);
          }
          h1 {
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
          }
          p.subtitle {
            margin-top: 0;
            color: #9ca3af;
            font-size: 0.9rem;
            margin-bottom: 1.2rem;
          }
          label {
            display: block;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
            color: #9ca3af;
          }
          input[type="text"],
          input[type="email"],
          textarea {
            width: 100%;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            border: 1px solid rgba(148,163,184,0.3);
            background: #020617;
            color: #e5e7eb;
            font-size: 0.9rem;
            box-sizing: border-box;
          }
          textarea {
            resize: vertical;
            min-height: 120px;
          }
          input:focus, textarea:focus {
            outline: none;
            border-color: #22c55e;
            box-shadow: 0 0 0 1px rgba(34,197,94,0.4);
          }
          .field {
            margin-bottom: 0.9rem;
          }
          .buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
          }
          button {
            border: none;
            border-radius: 999px;
            padding: 0.55rem 1.1rem;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
          }
          .btn-primary {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: #022c22;
          }
          .btn-primary:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
          }
          .btn-ghost {
            background: transparent;
            border: 1px solid rgba(148,163,184,0.4);
            color: #e5e7eb;
          }
          .btn-ghost:hover {
            border-color: #22c55e;
            color: #bbf7d0;
            transform: translateY(-1px);
          }
          .small-note {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.5rem;
          }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Student Helpdesk</h1>
          <p class="subtitle">
            Submit your issue or question. An administrator will review it and respond via email.
          </p>
          <form method="post" action="/public/ticket">
            <div class="field">
              <label for="name">Full name</label>
              <input type="text" id="name" name="name" required />
            </div>

            <div class="field">
              <label for="email">Email address</label>
              <input type="email" id="email" name="email" required />
            </div>

            <div class="field">
              <label for="subject">Subject</label>
              <input type="text" id="subject" name="subject" required />
            </div>

            <div class="field">
              <label for="body">Describe your issue</label>
              <textarea id="body" name="body" required></textarea>
            </div>

            <div class="buttons">
              <button type="submit" class="btn-primary">Submit</button>
            </div>
            <p class="small-note">
              By submitting, you agree that your request will be stored in the helpdesk system.
            </p>
          </form>
        </div>
      </body>
    </html>
    """


@app.post("/public/ticket", response_class=HTMLResponse)
async def submit_public_ticket(
    request: Request,
    name: str = Form(...),
    email: EmailStr = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Handle public ticket submission, run ML classification and routing,
    then redirect to thank-you page.
    """
    full_text = f"{subject}\n{body}"

    # ML category prediction
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    if PREDICTOR_LOADED:
        predicted_category, confidence = predict_category(full_text)

    # Auto priority
    priority_pred = compute_priority(full_text, predicted_category)
    priority_final = priority_pred

    # Auto queue routing
    category_final = predicted_category
    queue_value = route_to_queue(category_final)

    ticket = Ticket(
        name=name,
        email=email,
        subject=subject,
        body=body,
        category_pred=predicted_category,
        category_final=category_final,
        priority_pred=priority_pred,
        priority_final=priority_final,
        queue=queue_value,
        status="new",
        confidence=confidence,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    # Redirect to Thank-you page
    return RedirectResponse(
        url=f"/public/thanks?ticket_id={ticket.id}",
        status_code=303,
    )


@app.get("/public/thanks", response_class=HTMLResponse)
async def public_thanks(ticket_id: int):
    """
    Thank-you page after public ticket creation.
    """
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Ticket submitted</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
          }}
          .card {{
            background: #020617;
            border-radius: 16px;
            padding: 28px 34px;
            box-shadow: 0 22px 45px rgba(15,23,42,0.75);
            width: 100%;
            max-width: 520px;
            border: 1px solid rgba(148,163,184,0.2);
            text-align: center;
          }}
          h1 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
          }}
          p {{
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
          }}
          .ticket-id {{
            font-family: monospace;
            color: #a5b4fc;
          }}
          .buttons {{
            display: flex;
            justify-content: center;
            gap: 0.75rem;
            margin-top: 1.2rem;
          }}
          button {{
            border: none;
            border-radius: 999px;
            padding: 0.55rem 1.1rem;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
          }}
          .btn-primary {{
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: #022c22;
          }}
          .btn-primary:hover {{
            filter: brightness(1.05);
            transform: translateY(-1px);
          }}
          .btn-ghost {{
            background: transparent;
            border: 1px solid rgba(148,163,184,0.4);
            color: #e5e7eb;
          }}
          .btn-ghost:hover {{
            border-color: #22c55e;
            color: #bbf7d0;
            transform: translateY(-1px);
          }}
          .small-note {{
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 0.75rem;
          }}
        </style>
      </head>
      <body>
        <div class="card">
          <h1>Thank you for your request</h1>
          <p>Your ticket has been created successfully.</p>
          <p>Ticket ID: <span class="ticket-id">{ticket_id}</span></p>
          <p class="small-note">An administrator will review it and contact you by email if needed.</p>
          <div class="buttons">
            <button class="btn-primary"
                    onclick="window.location.href='/public/ticket'">
              Create new ticket
            </button>
            <button class="btn-ghost"
                    onclick="window.close()">
              Close page
            </button>
          </div>
        </div>
      </body>
    </html>
    """

# Admin CSV/JSON import (admin only)

@app.post("/admin/import-tickets")
async def import_tickets(
    file: Optional[UploadFile] = File(None),
    tickets: Optional[List[TicketCreate]] = Body(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
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

#Admin dataset + training + metrics

@app.get("/admin/dataset", response_class=HTMLResponse)
async def admin_dataset_page(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    total_samples = db.query(TrainingSample).count()
    last_run = db.query(TrainingRun).order_by(TrainingRun.run_at.desc()).first()

    if last_run:
        run_info = (
            f"Last run at {last_run.run_at}, "
            f"ML acc={last_run.accuracy_ml:.3f}, macro F1={last_run.macro_f1_ml:.3f}; "
            f"baseline acc={last_run.accuracy_baseline:.3f}, "
            f"macro F1={last_run.macro_f1_baseline:.3f}"
        )
    else:
        run_info = "No training runs yet."

    return f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8"/>
        <title>Training dataset – AI Helpdesk</title>
        <style>
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background:
              radial-gradient(circle at top left, #1e293b 0, #020617 45%),
              radial-gradient(circle at bottom right, #0f172a 0, #020617 55%);
            color: #e5e7eb;
          }}
          .card {{
            width: 760px;
            max-width: 96vw;
            background: rgba(15, 23, 42, 0.9);
            border-radius: 18px;
            padding: 30px 34px 28px;
            box-shadow:
              0 24px 60px rgba(15, 23, 42, 0.9),
              0 0 0 1px rgba(148, 163, 184, 0.35);
            backdrop-filter: blur(16px);
          }}
          .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 18px;
          }}
          .card-title {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }}
          .card-subtitle {{
            font-size: 13px;
            color: #9ca3af;
            margin-top: 4px;
          }}
          .nav-links a {{
            font-size: 13px;
            color: #a5b4fc;
            text-decoration: none;
            margin-left: 10px;
          }}
          .nav-links a:hover {{
            text-decoration: underline;
          }}
          .section-title {{
            font-size: 15px;
            font-weight: 600;
            margin: 18px 0 8px;
          }}
          .metric-row {{
            font-size: 13px;
            color: #d1d5db;
            margin-bottom: 2px;
          }}
          .metric-label {{
            color: #9ca3af;
          }}
          form {{
            margin-top: 10px;
          }}
          input[type="file"] {{
            font-size: 13px;
            color: #e5e7eb;
          }}
          .btn-primary,
          .btn-secondary {{
            border-radius: 999px;
            border: none;
            padding: 9px 18px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 8px;
          }}
          .btn-primary {{
            background-image: linear-gradient(90deg, #f97316, #22c55e);
            color: #111827;
            box-shadow:
              0 10px 25px rgba(34, 197, 94, 0.35),
              0 0 0 1px rgba(15, 23, 42, 0.9);
          }}
          .btn-primary:hover {{
            filter: brightness(1.05);
          }}
          .btn-secondary {{
            background: transparent;
            color: #e5e7eb;
            border: 1px solid rgba(148, 163, 184, 0.6);
          }}
          .btn-secondary:hover {{
            background: rgba(30, 64, 175, 0.5);
          }}
          .hint-text {{
            font-size: 12px;
            color: #9ca3af;
            margin-top: 4px;
          }}
          .metrics-link {{
            margin-top: 16px;
            font-size: 13px;
          }}
          .metrics-link a {{
            color: #a5b4fc;
            text-decoration: none;
          }}
          .metrics-link a:hover {{
            text-decoration: underline;
          }}
        </style>
      </head>
      <body>
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">Training dataset</div>
              <div class="card-subtitle">
                Manage labelled tickets used to train the classifier.
              </div>
            </div>
            <div class="nav-links">
              <a href="/tickets">Back to dashboard</a>
              <a href="/logout">Logout</a>
            </div>
          </div>

          <div>
            <div class="section-title">Dataset summary</div>
            <div class="metric-row">
              <span class="metric-label">Total labelled samples:</span>
              <span>{total_samples}</span>
            </div>
            <div class="metric-row">
              <span class="metric-label">Latest training run:</span>
              <span>{run_info}</span>
            </div>
          </div>

          <div>
            <div class="section-title">Upload labelled CSV</div>
            <div class="hint-text">
              CSV must have columns:
              <code>subject</code>, <code>body</code>, <code>true_category</code>,
              and optional <code>true_priority</code>.
            </div>
            <form method="post" action="/admin/dataset-upload" enctype="multipart/form-data">
              <input type="file" name="file" accept=".csv" required />
              <br/>
              <button type="submit" class="btn-primary">Upload CSV</button>
            </form>
          </div>

          <div>
            <div class="section-title">Train model</div>
            <div class="hint-text">
              Trains TF-IDF + Logistic Regression on all samples and logs metrics
              (including baseline keyword model).
            </div>
            <form method="post" action="/admin/train-model">
              <button type="submit" class="btn-secondary">Train from training_samples</button>
            </form>
          </div>

          <div class="metrics-link">
            <a href="/admin/metrics">View latest metrics report →</a>
          </div>
        </div>
      </body>
    </html>
    """
@app.post("/admin/dataset-upload")
async def admin_dataset_upload(
    file: UploadFile = File(...),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Upload labelled dataset.
    CSV headers: subject, body, true_category, true_priority
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded")

    reader = csv.DictReader(StringIO(text))
    required = {"subject", "body", "true_category"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain columns: subject, body, true_category",
        )

    imported = 0
    for row in reader:
        sample = TrainingSample(
            subject=row["subject"],
            body=row["body"],
            true_category=row["true_category"],
            true_priority=row.get("true_priority"),
        )
        db.add(sample)
        imported += 1

    db.commit()
    return RedirectResponse(
        url=f"/admin/dataset?imported={imported}",
        status_code=303,
    )


@app.post("/admin/train-model", response_class=HTMLResponse)
async def admin_train_model(
    current_user: User = Depends(require_admin),
):
    """
    Train model from training_samples.
    Calls train_and_save_from_db(), which writes metrics into training_runs.
    """
    metrics = train_and_save_from_db()
    if metrics is None:
        return HTMLResponse(
            "<h1>No training samples found.</h1>"
            "<p>Upload a labelled CSV on <a href='/admin/dataset'>/admin/dataset</a> first.</p>",
            status_code=400,
        )

    # After training, redirect to metrics page
    return RedirectResponse(url="/admin/metrics", status_code=303)


@app.get("/admin/metrics", response_class=HTMLResponse)
async def admin_metrics(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Show metrics of the latest training run:
    accuracy, precision, recall, F1 per category (for ML model).
    Also shows baseline summary.
    """
    run = db.query(TrainingRun).order_by(TrainingRun.run_at.desc()).first()
    if not run:
        return HTMLResponse(
            "<h1>No training runs yet</h1>"
            "<p>Go to <a href='/admin/dataset'>/admin/dataset</a> and train a model.</p>",
            status_code=200,
        )

    metrics = run.get_report()
    ml = metrics.get("ml", {})
    baseline = metrics.get("baseline", {})

    ml_accuracy = ml.get("accuracy", 0.0)
    ml_macro_f1 = ml.get("macro_f1", 0.0)
    base_accuracy = baseline.get("accuracy", 0.0)
    base_macro_f1 = baseline.get("macro_f1", 0.0)

    ml_report = ml.get("report", {})

    # Build per-category table for ML model
    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    rows = ""
    for label, stats in ml_report.items():
        if label in skip_keys:
            continue
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        f1 = stats.get("f1-score", 0.0)
        support = stats.get("support", 0)
        rows += (
            f"<tr><td>{label}</td>"
            f"<td>{prec:.2f}</td>"
            f"<td>{rec:.2f}</td>"
            f"<td>{f1:.2f}</td>"
            f"<td>{support}</td></tr>"
        )

    return f"""
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8"/><title>ML metrics</title></head>
      <body>
        <h1>Admin: Model metrics (latest run)</h1>
        <p><a href="/admin/dataset">Back to dataset</a> |
           <a href="/tickets">Dashboard</a> |
           <a href="/logout">Logout</a></p>

        <h3>Overall metrics</h3>
        <p><strong>ML accuracy:</strong> {ml_accuracy:.3f}</p>
        <p><strong>ML macro F1:</strong> {ml_macro_f1:.3f}</p>
        <p><strong>Baseline accuracy:</strong> {base_accuracy:.3f}</p>
        <p><strong>Baseline macro F1:</strong> {base_macro_f1:.3f}</p>

        <h3>Per-category metrics (ML model)</h3>
        <table border="1" cellpadding="4" cellspacing="0">
          <thead>
            <tr>
              <th>Category</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </body>
    </html>
    """


# Existing JSON API v1
app.include_router(tickets_router, prefix="/api/v1")
