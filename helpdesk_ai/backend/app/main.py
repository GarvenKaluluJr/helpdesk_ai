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



# =========================
# Phase 3.1 – seed admin
# =========================

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


# =========================
# Login / Logout
# =========================

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


# =========================
# Dashboard redirect → /tickets
# =========================

@app.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/tickets", status_code=302)


# =========================
# Helpers for display
# =========================

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


# =========================
# Phase 4.1 & 4.3 – Ticket list (HTML) + filters + pagination
# =========================

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
    """
    Agent dashboard: ticket list with filters & pagination.
    GET /tickets

    Filters:
      - category: applied on category_final OR (category_final is null and category_pred)
      - priority: priority_final
      - queue
      - status

    Pagination: page, page_size.
    """
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
            f"<td><a href='/tickets/{t.id}'>{t.id}</a></td>"
            f"<td>{t.subject}</td>"
            f"<td>{cat}</td>"
            f"<td>{prio}</td>"
            f"<td>{queue_str}</td>"
            f"<td>{t.status}</td>"
            f"<td>{created_str}</td>"
            f"</tr>"
        )

    # simple pagination links
    base_url = "/tickets"
    prev_link = ""
    next_link = ""
    if page > 1:
        prev_link = f"<a href='{base_url}?page={page-1}&page_size={page_size}'>Prev</a>"
    if page * page_size < total:
        next_link = f"<a href='{base_url}?page={page+1}&page_size={page_size}'>Next</a>"

    filters_info = (
        f"category={category or '*'}, "
        f"priority={priority or '*'}, "
        f"queue={queue or '*'}, "
        f"status={status or '*'}"
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
                <p>
          <a href="/logout">Logout</a> |
          <a href="/">Create ticket</a> |
          <a href="/admin/dataset">Training dataset</a>
        </p>
        <h3>Filters (current: {filters_info})</h3>
        <form method="get" action="/tickets">
          <label>Category:</label> <input type="text" name="category" value="{category or ''}" />
          <label>Priority:</label> <input type="text" name="priority" value="{priority or ''}" />
          <label>Queue:</label> <input type="text" name="queue" value="{queue or ''}" />
          <label>Status:</label> <input type="text" name="status" value="{status or ''}" />
          <button type="submit">Apply</button>
        </form>

        <p>Total tickets matching filters: {total}</p>

        <table border="1" cellpadding="4" cellspacing="0">
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

        <p>{prev_link} {next_link}</p>
      </body>
    </html>
    """


# =========================
# Phase 4.2 & 4.3 – Ticket detail (HTML)
# =========================

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

    return f"""
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8" /><title>Ticket {ticket.id}</title></head>
      <body>
        <h1>Ticket #{ticket.id}</h1>
        <p><a href="/tickets">Back to list</a> | <a href="/logout">Logout</a></p>

        <h3>Basic info</h3>
        <p><strong>Subject:</strong> {ticket.subject}</p>
        <p><strong>From:</strong> {ticket.name} &lt;{ticket.email}&gt;</p>
        <p><strong>Created at:</strong> {ticket.created_at}</p>
        <p><strong>Status:</strong> {ticket.status}</p>

        <h3>Message body</h3>
        <pre>{ticket.body}</pre>

        <h3>Category & Priority</h3>
        <p><strong>Category:</strong> {cat_display}</p>
        <p><strong>Priority:</strong> {prio_display}</p>
        <p><strong>Predicted category:</strong> {ticket.category_pred or '—'} (confidence {confidence_str})</p>
        <p><strong>Predicted priority:</strong> {ticket.priority_pred or '—'}</p>

        <h3>Manual edit</h3>
        <form method="post" action="/tickets/{ticket.id}/edit">
          <label>Final category:</label><br/>
          <input type="text" name="category_final" value="{category_final}" /><br/><br/>

          <label>Final priority:</label><br/>
          <input type="text" name="priority_final" value="{priority_final}" /><br/><br/>

          <label>Queue:</label><br/>
          <input type="text" name="queue" value="{queue_value}" /><br/><br/>

          <label>Status:</label><br/>
          <input type="text" name="status" value="{status_value}" /><br/><br/>

          <button type="submit">Save changes</button>
        </form>
      </body>
    </html>
    """


# =========================
# Phase 4.4 – HTML edit + ticket_history
# =========================

@app.post("/tickets/{ticket_id}/edit", response_class=HTMLResponse)
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

    changes = {}

    def maybe_update(field: str, new_val: Optional[str]):
        if new_val is None:
            return
        # treat empty string as clearing the field
        new_val_clean = new_val.strip() or None
        old_val = getattr(ticket, field)
        if old_val != new_val_clean:
            setattr(ticket, field, new_val_clean)
            changes[field] = (old_val, new_val_clean)

    maybe_update("category_final", category_final)
    maybe_update("priority_final", priority_final)
    maybe_update("queue", queue)
    if status:  # status shouldn't be cleared silently
        maybe_update("status", status)

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

    return RedirectResponse(url=f"/tickets/{ticket_id}", status_code=302)


# =========================
# Phase 4.4 – JSON PATCH /tickets/{id}
# =========================

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


# =========================
# Ticket creation form (still requires login)
# =========================

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
        <p><a href="/tickets">Back to dashboard</a> | <a href="/logout">Logout</a></p>
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

@app.post("/tickets", response_class=HTMLResponse)
async def submit_ticket(
    request: Request,
    name: str = Form(...),
    email: EmailStr = Form(...),
    subject: str = Form(...),
    body: str = Form(...),
    db: Session = Depends(get_db),
):
    full_text = f"{subject}\n{body}"

    # ML category prediction
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    if PREDICTOR_LOADED:
        predicted_category, confidence = predict_category(full_text)

    # Auto priority
    priority_pred = compute_priority(full_text, predicted_category)
    priority_final = priority_pred

    # Auto queue routing (Phase 7)
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

    html = f"""
    <html>
      <body>
        <h1>Ticket created successfully</h1>
        <p><strong>ID:</strong> {ticket.id}</p>
        <p><strong>Subject:</strong> {ticket.subject}</p>
        <p><strong>Category:</strong> {ticket.category_final or ticket.category_pred or "—"}</p>
        <p><strong>Priority:</strong> {ticket.priority_final or ticket.priority_pred or "—"}</p>
        <p><strong>Queue:</strong> {ticket.queue or "—"}</p>
        <p><a href="/tickets">Back to dashboard</a> | <a href="/">Create another ticket</a></p>
      </body>
    </html>
    """
    return HTMLResponse(html)


#====================
# Admin CSV/JSON import (admin only)
# =========================

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
# =========================
# Phase 8 – Admin dataset + training + metrics
# =========================

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
    <html>
      <head><meta charset="utf-8"/><title>Admin dataset</title></head>
      <body>
        <h1>Admin: Training dataset</h1>
        <p><a href="/tickets">Back to dashboard</a> | <a href="/logout">Logout</a></p>

        <p><strong>Total labelled samples:</strong> {total_samples}</p>
        <p><strong>Latest training run:</strong> {run_info}</p>

        <h3>Upload labelled CSV</h3>
        <p>CSV must have columns: subject, body, true_category, (optional) true_priority.</p>
        <form method="post" action="/admin/dataset-upload" enctype="multipart/form-data">
          <input type="file" name="file" accept=".csv" required />
          <button type="submit">Upload CSV</button>
        </form>

        <h3>Train model</h3>
        <form method="post" action="/admin/train-model">
          <button type="submit">Train from training_samples</button>
        </form>

        <p><a href="/admin/metrics">View metrics</a></p>
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
    Phase 8.1 upload labelled dataset.
    CSV headers: subject, body, true_category, (optional) true_priority
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
    Phase 8.2 train model from training_samples.
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
    Phase 8.3 – show metrics of the latest training run:
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

# =========================
# Existing JSON API v1
# =========================

app.include_router(tickets_router, prefix="/api/v1")
