from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declared_attr

from src.database.base import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    proposal = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    project_title = Column(String(255), nullable=True)
    timeline_weeks = Column(Integer, default=12)
    team_size = Column(Integer, default=5)
    tech_stack = Column(JSON, default=list)
    priority = Column(String(20), default="medium")
    diagram_types = Column(JSON, default=list)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @declared_attr
    def outputs(self):
        return None


class SessionOutput(Base):
    __tablename__ = "session_outputs"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    output_type = Column(String(50), nullable=False)
    output_data = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    is_valid = Column(Integer, default=1)
    feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = {"sqlite_autoincrement": True}

    def __init__(self, **kwargs):
        if "version" not in kwargs or kwargs.get("version") is None:
            kwargs["version"] = 1
        if "is_valid" not in kwargs or kwargs.get("is_valid") is None:
            kwargs["is_valid"] = 1
        super().__init__(**kwargs)


class TimeValidationSession(Base):
    __tablename__ = "time_validation_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), nullable=False, index=True)
    timetable = Column(Text, nullable=True)
    milestones = Column(JSON, default=list)
    parallel_work_streams = Column(JSON, default=list)
    validation_results = Column(JSON, nullable=True)
    is_valid = Column(Integer, default=0)
    feedback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)