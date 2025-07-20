"""
Database configuration and session management for PostgreSQL
Includes vector database setup with pgvector extension
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData, event, text
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
import asyncpg
from loguru import logger
from .config import settings

# Database engine and session
engine = None
async_session_maker = None

class Base(DeclarativeBase):
    """Base class for all database models"""
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s"
        }
    )

async def init_db():
    """Initialize database connection and create tables"""
    global engine, async_session_maker
    
    try:
        # Create async engine
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            poolclass=NullPool,  # Disable pooling for single-user deployment
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create session maker
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("✅ pgvector extension enabled")
        
        # Import models to register them
        from src.models.user import User
        from src.models.conversation import Conversation, Message
        from src.models.training import TrainingJob
        from src.models.analytics import Analytics
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    if not async_session_maker:
        raise RuntimeError("Database not initialized")
    
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def close_db():
    """Close database connections"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("✅ Database connections closed")

# Database health check
async def check_db_health() -> bool:
    """Check database connectivity"""
    try:
        if not engine:
            return False
            
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
