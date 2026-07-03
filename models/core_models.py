"""
Core data models for internal system operations
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from .api_models import SeverityLevel, AnalysisCategory, ReviewConfig, Finding, PRMetadata


class ChangeType(str, Enum):
    """Types of changes in a diff"""
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"


class LineChange(BaseModel):
    """Represents a single line change in a diff"""
    line_number: int = Field(gt=0)
    content: str
    change_type: ChangeType


class FileChange(BaseModel):
    """Represents changes to a single file"""
    file_path: str
    language: str
    is_binary: bool = False
    additions: List[LineChange] = Field(default_factory=list)
    deletions: List[LineChange] = Field(default_factory=list)
    modifications: List[LineChange] = Field(default_factory=list)

    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()


class ParsedDiff(BaseModel):
    """Parsed diff content with structured changes"""
    files: List[FileChange]
    total_additions: int = Field(default=0, ge=0)
    total_deletions: int = Field(default=0, ge=0)
    
    def model_post_init(self, __context):
        """Calculate totals after initialization"""
        self.total_additions = sum(len(f.additions) for f in self.files)
        self.total_deletions = sum(len(f.deletions) for f in self.files)


class ReviewContext(BaseModel):
    """Context information for agent analysis"""
    file_changes: List[FileChange]
    config: ReviewConfig
    pr_metadata: Optional[PRMetadata] = None


class AgentResult(BaseModel):
    """Result from a single analyzer agent"""
    agent_name: str
    findings: List[Finding]
    execution_time: float = Field(ge=0)  # seconds
    success: bool = True
    error_message: Optional[str] = None


class ReviewState(BaseModel):
    """State object for LangGraph workflow"""
    parsed_diff: Optional[ParsedDiff] = None
    config: Optional[ReviewConfig] = None
    pr_metadata: Optional[PRMetadata] = None
    agent_results: List[AgentResult] = Field(default_factory=list)
    final_findings: List[Finding] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Complete review result for storage"""
    review_id: str
    pr_metadata: PRMetadata
    commit_sha: str
    findings: List[Finding]
    timestamp: datetime
    config_used: ReviewConfig
    execution_time: float = Field(ge=0)  # seconds
    agent_results: List[AgentResult] = Field(default_factory=list)


class PRData(BaseModel):
    """Raw PR data from GitHub API"""
    metadata: PRMetadata
    diff_content: str
    files_changed: List[str] = Field(default_factory=list)


class Comment(BaseModel):
    """Formatted comment for output"""
    file_path: str
    line_number: int
    content: str  # Markdown formatted content
    findings: List[Finding]


class FormattedReview(BaseModel):
    """Formatted review output"""
    markdown_output: str
    structured_comments: List[Comment]
    summary: Dict[str, Any]