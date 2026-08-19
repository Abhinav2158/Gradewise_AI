from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.db.models import Base
from src import config

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes all database tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

def get_db() -> Session:
    """Dependency / context manager generator for database sessions."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass
