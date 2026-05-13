from contextlib import contextmanager

from sqlalchemy import QueuePool, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session


from src.config import Config

Base = declarative_base()

_engine = None
_SessionLocal = None

def get_engine():
    global _engine

    if _engine is None:
        _engine = create_engine(
            Config.DATABASE_URL,
            echo=False,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    return _engine

def get_session_local():
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=get_engine(),
            )
        )

    return _SessionLocal

SessionLocal = get_session_local()


@contextmanager
def get_db():
    db = get_session_local()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        get_session_local().remove()


def init_db():
    Base.metadata.create_all(bind=get_engine())


def close_db():
    get_session_local().remove()
    get_engine().dispose()