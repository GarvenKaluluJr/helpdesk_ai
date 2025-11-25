# backend/app/models/queue.py
from sqlalchemy import Column, Integer, String

from ..db import Base


class Queue(Base):
    __tablename__ = "queues"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
