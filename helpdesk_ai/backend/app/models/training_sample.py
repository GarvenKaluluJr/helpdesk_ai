# helpdesk_ai/backend/app/models/training_sample.py
from sqlalchemy import Column, Integer, Text, DateTime, func

from ..db import Base


class TrainingSample(Base):
    """
    Labelled tickets used for ML training.
    Comes from /admin/dataset-upload CSV.
    """
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    true_category = Column(Text, nullable=False)
    true_priority = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
