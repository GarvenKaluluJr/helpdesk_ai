# backend/app/main.py
from fastapi import FastAPI

from .api.v1.tickets import router as tickets_router

app = FastAPI(title="AI Helpdesk Ticket Classifier")


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(tickets_router, prefix="/api/v1")
