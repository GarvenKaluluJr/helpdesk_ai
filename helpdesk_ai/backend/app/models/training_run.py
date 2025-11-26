# helpdesk_ai/backend/app/models/training_run.py
import json
from sqlalchemy import Column, Integer, DateTime, Float, Text, func

from ..db import Base


class TrainingRun(Base):
    """
    Stores summary metrics for each training run (ML vs baseline).
    report_json is a JSON string with per class metrics, etc.
    """
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_at = Column(DateTime(timezone=True), server_default=func.now())

    # ML model
    accuracy_ml = Column(Float, nullable=True)
    macro_f1_ml = Column(Float, nullable=True)

    # Keyword baseline
    accuracy_baseline = Column(Float, nullable=True)
    macro_f1_baseline = Column(Float, nullable=True)

    # Full metrics (classification_report, etc.) as JSON text
    report_json = Column(Text, nullable=True)

    def set_report(self, data: dict) -> None:
        self.report_json = json.dumps(data)

    def get_report(self) -> dict:
        if not self.report_json:
            return {}
        return json.loads(self.report_json)
