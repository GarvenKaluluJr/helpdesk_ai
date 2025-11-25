# backend/app/models/ticket_history.py

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..db import Base


class TicketHistory(Base):
    __tablename__ = "ticket_history"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)
    field = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=False)
    changed_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    changed_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )

    ticket = relationship("Ticket", back_populates="history_entries")
    # changed_by_user = relationship("User")  # optional, not needed for Phase 1
