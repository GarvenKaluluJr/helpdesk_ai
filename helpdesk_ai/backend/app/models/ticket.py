# backend/app/models/ticket.py

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..db import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)

    category_pred = Column(String(100), nullable=True)
    category_final = Column(String(100), nullable=True)

    priority_pred = Column(String(50), nullable=True)
    priority_final = Column(String(50), nullable=True)

    queue = Column(String(100), nullable=True)  # queue name (IT, Finance, etc.)

    status = Column(String(50), nullable=False, server_default="new")

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    confidence = Column(Float, nullable=True)

    history_entries = relationship("TicketHistory", back_populates="ticket")
