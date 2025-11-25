# backend/app/models/__init__.py

from .user import User
from .queue import Queue
from .category import Category
from .ticket import Ticket
from .ticket_history import TicketHistory

__all__ = ["User", "Queue", "Category", "Ticket", "TicketHistory"]
