"""
Pydantic models for API requests and responses
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class SeverityLevel(str, Enum):
    """Severity levels for findings"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisCategory(str, Enum):
    """Categories of code analysis"""
    LOGIC = "logic"
    READABILITY = "readability"
    PERFORMANCE = "performance"
    SECURITY = "security"


class ReviewConfig(BaseModel):
    """Configuration for review behavior"""
    severity_threshold: Optional[SeverityLevel] = SeverityLevel.LOW
    enabled_categories: List[AnalysisCategory] = Field(
        default_factory=lambda: [
            AnalysisCategory.LOGIC,
            AnalysisCategory.READABILITY,
            AnalysisCategory.PERFORMANCE,
            AnalysisCategory.SECURITY
        ]
    )
    custom_rules: Optional[Dict[str, Any]] = None

    @validator('enabled_categories')
    def validate_categories(cls, v):
        if not v:
            raise ValueError("At least one analysis category must be enabled")
        return v


class ReviewRequest(BaseModel):
    """Request model for creating a review"""
    pr_url: Optional[str] = None
    repository: Optional[str] = None
    pr_number: Optional[int] = None
    diff_content: Optional[str] = None
    github_token: Optional[str] = None
    config: Optional[ReviewConfig] = None

    @validator('pr_number')
    def validate_pr_number(cls, v):
        if v is not None and v <= 0:
            raise ValueError("PR number must be positive")
        return v

    def model_post_init(self, __context):
        """Validate that either PR URL/details or diff content is provided"""
        has_pr_info = bool(self.pr_url or (self.repository and self.pr_number))
        has_diff = bool(self.diff_content)
        
        if not (has_pr_info or has_diff):
            raise ValueError(
                "Either pr_url (or repository + pr_number) or diff_content must be provided"
            )


class PRMetadata(BaseModel):
    """Metadata about a Pull Request"""
    repository: str
    pr_number: int
    title: str
    author: str
    commit_sha: str
    base_branch: str
    head_branch: str


class Finding(BaseModel):
    """A single code review finding"""
    file_path: str
    line_number: int
    severity: SeverityLevel
    category: AnalysisCategory
    description: str
    suggestion: Optional[str] = None
    agent_source: str

    @validator('line_number')
    def validate_line_number(cls, v):
        if v <= 0:
            raise ValueError("Line number must be positive")
        return v

    @validator('description')
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()
    
    def dict(self, **kwargs):
        """Override dict to include message field for frontend compatibility"""
        d = super().dict(**kwargs)
        d['message'] = d['description']  # Add message as alias
        return d
    
    def model_dump(self, **kwargs):
        """Override model_dump for Pydantic v2 compatibility"""
        d = super().model_dump(**kwargs)
        d['message'] = d['description']  # Add message as alias
        return d


class ReviewSummary(BaseModel):
    """Summary statistics for a review"""
    total_findings: int = Field(ge=0)
    findings_by_severity: Dict[SeverityLevel, int] = Field(default_factory=dict)
    findings_by_category: Dict[AnalysisCategory, int] = Field(default_factory=dict)
    files_analyzed: int = Field(ge=0)
    lines_changed: int = Field(ge=0)


class ReviewResponse(BaseModel):
    """Response model for review results"""
    review_id: str
    pr_metadata: PRMetadata
    findings: List[Finding]
    summary: ReviewSummary
    timestamp: datetime
    config_used: Optional[ReviewConfig] = None


class ReviewStatus(BaseModel):
    """Status of a review in progress"""
    review_id: str
    status: str  # "in_progress", "completed", "failed"
    progress: Optional[str] = None
    error_message: Optional[str] = None


class HistoryFilters(BaseModel):
    """Filters for querying review history"""
    repository: Optional[str] = None
    pr_number: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    severity: Optional[SeverityLevel] = None
    category: Optional[AnalysisCategory] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ErrorResponse(BaseModel):
    """Error response model"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)