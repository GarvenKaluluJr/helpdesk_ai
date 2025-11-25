# backend/app/db.py
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load DATABASE_URL from .env at project root
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment/.env")

engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

Base = declarative_base()


def get_db():
    """FastAPI dependency to provide DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
