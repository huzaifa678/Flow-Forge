"""initial

Revision ID: aa4edf8d1804
Revises: 
Create Date: 2026-05-13 15:54:43.924257

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aa4edf8d1804'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("proposal", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("project_title", sa.String(length=255), nullable=True),
        sa.Column("timeline_weeks", sa.Integer(), nullable=True),
        sa.Column("team_size", sa.Integer(), nullable=True),
        sa.Column("tech_stack", sa.JSON(), nullable=True),
        sa.Column("priority", sa.String(length=20), nullable=True),
        sa.Column("diagram_types", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sessions_session_id", "sessions", ["session_id"], unique=True)

    op.create_table(
        "session_outputs",
        sa.Column("id", sa.Integer(), sa.Identity(always=False), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("output_type", sa.String(length=50), nullable=False),
        sa.Column("output_data", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=True),
        sa.Column("is_valid", sa.Integer(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_session_outputs_session_id", "session_outputs", ["session_id"], unique=False)

    op.create_table(
        "time_validation_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("timetable", sa.Text(), nullable=True),
        sa.Column("milestones", sa.JSON(), nullable=True),
        sa.Column("parallel_work_streams", sa.JSON(), nullable=True),
        sa.Column("validation_results", sa.JSON(), nullable=True),
        sa.Column("is_valid", sa.Integer(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_time_validation_sessions_session_id", "time_validation_sessions", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_time_validation_sessions_session_id", table_name="time_validation_sessions")
    op.drop_table("time_validation_sessions")
    op.drop_index("ix_session_outputs_session_id", table_name="session_outputs")
    op.drop_table("session_outputs")
    op.drop_index("ix_sessions_session_id", table_name="sessions")
    op.drop_table("sessions")
