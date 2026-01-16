"""Database models and connection for job persistence."""
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# Database file path
DB_PATH = "jobs.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Job(Base):
    """Job model for tracking transcription jobs.
    
    Note: API keys are NEVER stored in the database for security.
    They are passed directly to the worker and only kept in memory.
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False)  # queued, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    audio_filename = Column(String, nullable=False)
    diarize = Column(Boolean, default=False, nullable=False)
    error_message = Column(String, nullable=True)
    transcript_filename = Column(String, nullable=True)


def init_db():
    """Initialize the database, creating tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
