"""
Database configuration and connection management
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.pool import NullPool

from config import settings
from models.database import Base


class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        self.engine = None
        self.session_factory = None
        
    def initialize(self):
        """Initialize database engine and session factory"""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Set to True for SQL logging in development
            poolclass=NullPool if "sqlite" in self.database_url else None,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def create_tables(self):
        """Create all database tables"""
        if not self.engine:
            self.initialize()
            
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all database tables (for testing)"""
        if not self.engine:
            self.initialize()
            
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup"""
        if not self.session_factory:
            self.initialize()
            
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


# Convenience function for getting sessions
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session - convenience function"""
    async with db_manager.get_session() as session:
        yield session


# Database dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    async with db_manager.get_session() as session:
        yield session