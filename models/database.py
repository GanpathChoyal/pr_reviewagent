"""
SQLAlchemy database models for PR Review Agent
"""
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column, String, Integer, DateTime, Text, Boolean, 
    ForeignKey, Index, JSON, Float
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func

Base = declarative_base()


class Review(Base):
    """Database model for review records"""
    __tablename__ = "reviews"
    
    # Primary key
    review_id: Mapped[str] = Column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # PR metadata
    repository: Mapped[str] = Column(String(255), nullable=False)
    pr_number: Mapped[int] = Column(Integer, nullable=False)
    pr_title: Mapped[str] = Column(String(500), nullable=False)
    pr_author: Mapped[str] = Column(String(100), nullable=False)
    commit_sha: Mapped[str] = Column(String(40), nullable=False)
    base_branch: Mapped[str] = Column(String(100), nullable=False)
    head_branch: Mapped[str] = Column(String(100), nullable=False)
    
    # Review metadata
    timestamp: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    config: Mapped[dict] = Column(JSON, nullable=True)
    summary: Mapped[dict] = Column(JSON, nullable=True)
    execution_time: Mapped[float] = Column(Float, nullable=False, default=0.0)
    
    # Audit fields
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationships
    findings: Mapped[List["Finding"]] = relationship(
        "Finding", 
        back_populates="review",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_repo_pr', 'repository', 'pr_number'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_commit_sha', 'commit_sha'),
        Index('idx_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Review(id={self.review_id}, repo={self.repository}, pr={self.pr_number})>"


class Finding(Base):
    """Database model for individual findings"""
    __tablename__ = "findings"
    
    # Primary key
    finding_id: Mapped[str] = Column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to review
    review_id: Mapped[str] = Column(
        UUID(as_uuid=False), 
        ForeignKey('reviews.review_id', ondelete='CASCADE'),
        nullable=False
    )
    
    # Finding details
    file_path: Mapped[str] = Column(String(500), nullable=False)
    line_number: Mapped[int] = Column(Integer, nullable=False)
    severity: Mapped[str] = Column(String(20), nullable=False)
    category: Mapped[str] = Column(String(50), nullable=False)
    description: Mapped[str] = Column(Text, nullable=False)
    suggestion: Mapped[Optional[str]] = Column(Text, nullable=True)
    agent_source: Mapped[str] = Column(String(50), nullable=False)
    
    # Audit fields
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    
    # Relationships
    review: Mapped["Review"] = relationship("Review", back_populates="findings")
    
    # Indexes
    __table_args__ = (
        Index('idx_review_id', 'review_id'),
        Index('idx_severity', 'severity'),
        Index('idx_category', 'category'),
        Index('idx_file_path', 'file_path'),
        Index('idx_agent_source', 'agent_source'),
    )
    
    def __repr__(self):
        return f"<Finding(id={self.finding_id}, file={self.file_path}, line={self.line_number})>"


class AgentExecution(Base):
    """Database model for tracking agent execution details"""
    __tablename__ = "agent_executions"
    
    # Primary key
    execution_id: Mapped[str] = Column(
        UUID(as_uuid=False), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to review
    review_id: Mapped[str] = Column(
        UUID(as_uuid=False), 
        ForeignKey('reviews.review_id', ondelete='CASCADE'),
        nullable=False
    )
    
    # Agent execution details
    agent_name: Mapped[str] = Column(String(50), nullable=False)
    execution_time: Mapped[float] = Column(Float, nullable=False)
    success: Mapped[bool] = Column(Boolean, nullable=False, default=True)
    error_message: Mapped[Optional[str]] = Column(Text, nullable=True)
    findings_count: Mapped[int] = Column(Integer, nullable=False, default=0)
    
    # Audit fields
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    
    # Relationships
    review: Mapped["Review"] = relationship("Review")
    
    # Indexes
    __table_args__ = (
        Index('idx_agent_review', 'review_id', 'agent_name'),
        Index('idx_agent_name', 'agent_name'),
        Index('idx_success', 'success'),
    )
    
    def __repr__(self):
        return f"<AgentExecution(agent={self.agent_name}, success={self.success})>"