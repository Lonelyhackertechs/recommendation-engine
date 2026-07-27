import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

load_dotenv()

# Connection pool configuration
_engine = None

def get_connection():
    """Return a reusable connection pool for database access."""
    global _engine
    
    if _engine is None:
        database_url = os.getenv("DATABASE_URL")
        _engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600    # Recycle connections every hour
        )
    
    return _engine
