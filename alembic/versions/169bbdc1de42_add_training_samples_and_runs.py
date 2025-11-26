"""add training samples and runs

Revision ID: 169bbdc1de42
Revises: 0001_initial_schema
Create Date: 2025-11-26 13:59:24.001257
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "169bbdc1de42"
down_revision: Union[str, Sequence[str], None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: ADD training tables (do NOT drop existing ones)."""

    # Table to store labelled training samples
    op.create_table(
        "training_samples",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("true_category", sa.Text(), nullable=False),
        sa.Column("true_priority", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # Table to store training runs + metrics
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column(
            "run_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("accuracy_ml", sa.Float(), nullable=True),
        sa.Column("macro_f1_ml", sa.Float(), nullable=True),
        sa.Column("accuracy_baseline", sa.Float(), nullable=True),
        sa.Column("macro_f1_baseline", sa.Float(), nullable=True),
        sa.Column("report_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema: DROP training tables only."""
    op.drop_table("training_runs")
    op.drop_table("training_samples")
