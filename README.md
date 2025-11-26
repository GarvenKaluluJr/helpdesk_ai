# AI Helpdesk Ticket Classifier

University IT helpdesk web app with automatic ticket categorisation, priority assignment, and routing to department queues.  
Built with FastAPI + PostgreSQL + scikit-learn.

## Features

- Ticket creation form (subject + body, optional category hint)
- User auth with roles:
  - `admin` (full access, admin pages)
- Agent dashboard:
  - Ticket list with filters (category, priority, queue, status) + pagination
  - Ticket detail view + manual edit (category_final, priority_final, queue, status)
- ML classification (scikit-learn):
  - TF-IDF + Logistic Regression text classifier
  - Predicted category + confidence stored in DB
  - Simple rules-based priority (Low/Medium/High)
  - Automatic queue routing (IT, Finance, Admissions, General)
- Admin / research interface:
  - Upload labelled CSV training dataset
  - Train model from DB (`training_samples` → `ticket_category_model.joblib`)
  - Metrics page with ML vs baseline (keyword rules) comparison
- History & reliability:
  - `ticket_history` table logging manual changes
  - Basic load test script
  - Simple DB backup script

## Tech stack

- Python 3.9
- FastAPI, Uvicorn
- PostgreSQL, SQLAlchemy, Alembic
- Pydantic
- scikit-learn, joblib
- pytest

## Setup

```bash
git clone <this-repo-url>
cd helpdesk_ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
Configure postgres(database)
export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/helpdesk_ai
alembic upgrade head

Train initial model
python -m helpdesk_ai.backend.app.ml.train_classifier

run the app
python -m uvicorn helpdesk_ai.backend.app.main:app --reload

App will be available at:

Login: http://127.0.0.1:8000/login

Swagger UI (JSON API): http://127.0.0.1:8000/docs

Default admin user (seeded on startup):

username: admin

password: admin123

Usage (quick tour)

Log in as admin → redirected to /tickets (Ticket Dashboard).

Create tickets via / (Create Ticket) or API.

New tickets automatically get:

predicted category + confidence

priority (Low/Medium/High)

queue (IT / Finance / Admissions / General)

Admin pages:

/admin/dataset – upload labelled CSV + trigger training

/admin/metrics – view latest ML vs baseline metrics


Run tests
python -m pytest

Simple concurrency/load check:
python scripts/load_test.py

Database backup:
./scripts/backup_db.sh


Good luck!
