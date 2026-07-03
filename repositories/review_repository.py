"""
Review Repository for persisting and retrieving review results
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import select, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.database import Review as ReviewModel, Finding as FindingModel
from models import Finding, ReviewSummary, PRMetadata, ReviewConfig, SeverityLevel, AnalysisCategory

logger = logging.getLogger(__name__)


class ReviewResult:
    """Review result with all associated data"""
    
    def __init__(
        self,
        review_id: str,
        pr_metadata: PRMetadata,
        commit_sha: str,
        findings: List[Finding],
        timestamp: datetime,
        config_used: ReviewConfig
    ):
        self.review_id = review_id
        self.pr_metadata = pr_metadata
        self.commit_sha = commit_sha
        self.findings = findings
        self.timestamp = timestamp
        self.config_used = config_used


class HistoryFilters:
    """Filters for querying review history"""
    
    def __init__(
        self,
        repository: Optional[str] = None,
        pr_number: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        self.repository = repository
        self.pr_number = pr_number
        self.start_date = start_date
        self.end_date = end_date
        self.severity = severity
        self.category = category
        self.limit = limit
        self.offset = offset


class ReviewRepository:
    """Repository for review data persistence"""
    
    def __init__(self, session: AsyncSession):
        """
        Initialize Review Repository
        
        Args:
            session: Async database session
        """
        self.session = session
        logger.debug("ReviewRepository initialized")
    
    async def save_review(
        self,
        pr_metadata: PRMetadata,
        commit_sha: str,
        findings: List[Finding],
        config: ReviewConfig,
        summary: Optional[ReviewSummary] = None,
        execution_time: float = 0.0
    ) -> str:
        """
        Save review results to database
        
        Args:
            pr_metadata: PR metadata
            commit_sha: Commit SHA
            findings: List of findings
            config: Review configuration used
            summary: Optional review summary
            execution_time: Time taken to perform the review in seconds
            
        Returns:
            Review ID (UUID string)
        """
        try:
            # Generate review ID
            review_id = str(uuid4())
            
            # Create review model
            review = ReviewModel(
                review_id=review_id,
                repository=pr_metadata.repository,
                pr_number=pr_metadata.pr_number,
                pr_title=pr_metadata.title or "",
                pr_author=pr_metadata.author or "",
                commit_sha=commit_sha,
                base_branch=pr_metadata.base_branch or "",
                head_branch=pr_metadata.head_branch or "",
                timestamp=datetime.utcnow(),
                execution_time=execution_time,
                config={
                    "severity_threshold": config.severity_threshold,
                    "enabled_categories": config.enabled_categories,
                    "custom_rules": config.custom_rules
                },
                summary={
                    "total_findings": len(findings),
                    "findings_by_severity": self._count_by_severity(findings),
                    "findings_by_category": self._count_by_category(findings),
                    "files_analyzed": len(set(f.file_path for f in findings))
                } if not summary else {
                    "total_findings": summary.total_findings,
                    "findings_by_severity": summary.findings_by_severity,
                    "findings_by_category": summary.findings_by_category,
                    "files_analyzed": summary.files_analyzed,
                    "lines_changed": summary.lines_changed
                }
            )
            
            # Add review to session
            self.session.add(review)
            
            # Create finding models
            for finding in findings:
                finding_model = FindingModel(
                    finding_id=str(uuid4()),
                    review_id=review_id,
                    file_path=finding.file_path,
                    line_number=finding.line_number,
                    severity=finding.severity.value,
                    category=finding.category.value,
                    description=finding.description,
                    suggestion=finding.suggestion,
                    agent_source=finding.agent_source
                )
                self.session.add(finding_model)
            
            # Commit transaction
            await self.session.commit()
            
            logger.info(
                f"Saved review {review_id} with {len(findings)} findings "
                f"for {pr_metadata.repository}#{pr_metadata.pr_number}"
            )
            
            return review_id
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to save review: {e}", exc_info=True)
            raise
    
    async def get_review(self, review_id: str) -> Optional[ReviewResult]:
        """
        Retrieve review by ID
        
        Args:
            review_id: Review UUID
            
        Returns:
            ReviewResult or None if not found
        """
        try:
            # Query review with findings
            stmt = (
                select(ReviewModel)
                .options(selectinload(ReviewModel.findings))
                .where(ReviewModel.review_id == review_id)
            )
            
            result = await self.session.execute(stmt)
            review_model = result.scalar_one_or_none()
            
            if not review_model:
                logger.warning(f"Review {review_id} not found")
                return None
            
            # Convert to ReviewResult
            review_result = self._model_to_result(review_model)
            
            logger.debug(f"Retrieved review {review_id}")
            
            return review_result
            
        except Exception as e:
            logger.error(f"Failed to get review {review_id}: {e}", exc_info=True)
            return None
    
    async def query_reviews(self, filters: HistoryFilters) -> List[ReviewResult]:
        """
        Query reviews with filtering
        
        Args:
            filters: Query filters
            
        Returns:
            List of ReviewResult objects
        """
        try:
            # Build query
            stmt = select(ReviewModel).options(selectinload(ReviewModel.findings))
            
            # Apply filters
            conditions = []
            
            if filters.repository:
                conditions.append(ReviewModel.repository == filters.repository)
            
            if filters.pr_number is not None:
                conditions.append(ReviewModel.pr_number == filters.pr_number)
            
            if filters.start_date:
                conditions.append(ReviewModel.timestamp >= filters.start_date)
            
            if filters.end_date:
                conditions.append(ReviewModel.timestamp <= filters.end_date)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            # Order by timestamp descending
            stmt = stmt.order_by(desc(ReviewModel.timestamp))
            
            # Apply pagination
            stmt = stmt.limit(filters.limit).offset(filters.offset)
            
            # Execute query
            result = await self.session.execute(stmt)
            review_models = result.scalars().all()
            
            # Convert to ReviewResult objects
            reviews = [self._model_to_result(model) for model in review_models]
            
            # Apply finding-level filters if specified
            if filters.severity or filters.category:
                reviews = self._filter_by_findings(reviews, filters)
            
            logger.debug(f"Query returned {len(reviews)} reviews")
            
            return reviews
            
        except Exception as e:
            logger.error(f"Failed to query reviews: {e}", exc_info=True)
            return []
    
    async def get_reviews_by_pr(
        self,
        repository: str,
        pr_number: int
    ) -> List[ReviewResult]:
        """
        Get all reviews for a specific PR
        
        Args:
            repository: Repository name
            pr_number: PR number
            
        Returns:
            List of ReviewResult objects ordered by timestamp
        """
        filters = HistoryFilters(
            repository=repository,
            pr_number=pr_number,
            limit=1000  # High limit for PR history
        )
        
        return await self.query_reviews(filters)
    
    def _model_to_result(self, model: ReviewModel) -> ReviewResult:
        """Convert database model to ReviewResult"""
        # Convert findings
        findings = []
        for finding_model in model.findings:
            finding = Finding(
                file_path=finding_model.file_path,
                line_number=finding_model.line_number,
                severity=SeverityLevel(finding_model.severity),
                category=AnalysisCategory(finding_model.category),
                description=finding_model.description,
                suggestion=finding_model.suggestion,
                agent_source=finding_model.agent_source
            )
            findings.append(finding)
        
        # Create PR metadata
        pr_metadata = PRMetadata(
            repository=model.repository,
            pr_number=model.pr_number,
            title="",  # Not stored in review model
            author="",  # Not stored in review model
            commit_sha=model.commit_sha,
            base_branch="",  # Not stored in review model
            head_branch=""  # Not stored in review model
        )
        
        # Create config
        config_data = model.config or {}
        config = ReviewConfig(
            severity_threshold=config_data.get("severity_threshold"),
            enabled_categories=config_data.get("enabled_categories", [
                "logic", "readability", "performance", "security"
            ]),
            custom_rules=config_data.get("custom_rules")
        )
        
        return ReviewResult(
            review_id=model.review_id,
            pr_metadata=pr_metadata,
            commit_sha=model.commit_sha,
            findings=findings,
            timestamp=model.timestamp,
            config_used=config
        )
    
    def _filter_by_findings(
        self,
        reviews: List[ReviewResult],
        filters: HistoryFilters
    ) -> List[ReviewResult]:
        """Filter reviews based on finding-level criteria"""
        filtered = []
        
        for review in reviews:
            # Check if review has findings matching criteria
            matching_findings = review.findings
            
            if filters.severity:
                matching_findings = [
                    f for f in matching_findings
                    if f.severity.value.lower() == filters.severity.lower()
                ]
            
            if filters.category:
                matching_findings = [
                    f for f in matching_findings
                    if f.category.value.lower() == filters.category.lower()
                ]
            
            # Include review if it has matching findings
            if matching_findings:
                filtered.append(review)
        
        return filtered
    
    def _count_by_severity(self, findings: List[Finding]) -> Dict[str, int]:
        """Count findings by severity"""
        counts = {}
        for finding in findings:
            severity = finding.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _count_by_category(self, findings: List[Finding]) -> Dict[str, int]:
        """Count findings by category"""
        counts = {}
        for finding in findings:
            category = finding.category.value
            counts[category] = counts.get(category, 0) + 1
        return counts