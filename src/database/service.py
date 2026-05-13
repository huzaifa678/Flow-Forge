from contextlib import contextmanager
from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

from src.config import Config
from src.database.models import Base, Session as SessionModel, SessionOutput

engine = create_engine(
    Config.DATABASE_URL,
    echo=False,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        SessionLocal.remove()


def init_db():
    Base.metadata.create_all(bind=engine)


def close_db():
    SessionLocal.remove()
    engine.dispose()


class SessionService:
    @staticmethod
    def create_session(
        proposal: str,
        prompt: str,
        project_title: str = "",
        timeline_weeks: int = 12,
        team_size: int = 5,
        tech_stack: list[str] | None = None,
        priority: str = "medium",
        diagram_types: list[str] | None = None,
    ) -> str:
        import uuid

        session_id = str(uuid.uuid4())[:16]

        with get_db() as db:
            session = SessionModel(
                session_id=session_id,
                proposal=proposal,
                prompt=prompt,
                project_title=project_title or proposal[:100],
                timeline_weeks=timeline_weeks,
                team_size=team_size,
                tech_stack=tech_stack or [],
                priority=priority,
                diagram_types=diagram_types or ["workflow", "ci_cd"],
            )
            db.add(session)

        return session_id

    @staticmethod
    def save_output(
        session_id: str,
        agent_name: str,
        output_type: str,
        output_data: dict[str, Any],
        feedback: str | None = None,
        is_valid: bool = True,
        version: int = 1,
    ) -> int:
        with get_db() as db:
            output = SessionOutput(
                session_id=session_id,
                agent_name=agent_name,
                output_type=output_type,
                output_data=output_data,
                feedback=feedback,
                is_valid=1 if is_valid else 0,
                version=version,
            )
            db.add(output)
            db.flush()
            return output.id

    @staticmethod
    def get_outputs(
        session_id: str,
        agent_name: Optional[str] = None,
        output_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with get_db() as db:
            query = db.query(SessionOutput).filter(
                SessionOutput.session_id == session_id
            )

            if agent_name:
                query = query.filter(SessionOutput.agent_name == agent_name)
            if output_type:
                query = query.filter(SessionOutput.output_type == output_type)

            return [
                {
                    "id": o.id,
                    "session_id": o.session_id,
                    "agent_name": o.agent_name,
                    "output_type": o.output_type,
                    "output_data": o.output_data,
                    "feedback": o.feedback,
                    "is_valid": bool(o.is_valid),
                    "version": o.version,
                    "created_at": o.created_at,
                }
                for o in query.order_by(SessionOutput.version.desc()).all()
            ]

    @staticmethod
    def get_latest_output(
        session_id: str,
        agent_name: str,
        output_type: str,
    ) -> Optional[dict[str, Any]]:
        outputs = SessionService.get_outputs(session_id, agent_name, output_type)
        return outputs[0] if outputs else None

    @staticmethod
    def get_previous_invalid_output(
        session_id: str,
        agent_name: str,
        output_type: str,
    ) -> Optional[dict[str, Any]]:
        with get_db() as db:
            output = (
                db.query(SessionOutput)
                .filter(
                    SessionOutput.session_id == session_id,
                    SessionOutput.agent_name == agent_name,
                    SessionOutput.output_type == output_type,
                    SessionOutput.is_valid == 0,
                )
                .order_by(SessionOutput.created_at.desc())
                .first()
            )

            if output:
                return {
                    "id": output.id,
                    "output_data": output.output_data,
                    "feedback": output.feedback,
                    "version": output.version,
                }
            return None