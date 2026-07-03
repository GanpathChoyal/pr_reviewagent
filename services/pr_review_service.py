"""
PR Review Service - Main coordinator for the review workflow
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel

from services.github_client import GitHubAPIClient, GitHubAPIError
from services.diff_parser import DiffParser, DiffParseError
from services.review_orchestrator import ReviewOrchestrator
from services.comment_generator import CommentGenerator, FormattedReview
from repositories.review_repository import ReviewRepository
from models import (
    ReviewConfig,
    PRMetadata,
    ReviewResponse,
    ParsedDiff
)

logger = logging.getLogger(__name__)


class PRReviewService:
    """Main service for coordinating PR review workflow"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        github_token: Optional[str] = None,
        db_session: Optional[Any] = None
    ):
        """
        Initialize PR Review Service
        
        Args:
            llm: Language model for analyzer agents
            github_token: GitHub API token (optional)
            db_session: Database session for persistence (optional)
        """
        self.llm = llm
        self.github_client = GitHubAPIClient(token=github_token)
        self.diff_parser = DiffParser()
        self.comment_generator = CommentGenerator()
        self.db_session = db_session
        
        logger.info("PRReviewService initialized")
    
    async def review_pr_from_url(
        self,
        pr_url: str,
        config: Optional[ReviewConfig] = None
    ) -> ReviewResponse:
        """
        Review a PR from GitHub URL
        
        Args:
            pr_url: GitHub PR URL
            config: Review configuration
            
        Returns:
            ReviewResponse with findings and formatted comments
        """
        try:
            if not self.github_client:
                raise ValueError("GitHub client not initialized - token required for PR URL reviews")
            
            logger.info(f"Starting review for PR: {pr_url}")
            start_time = datetime.utcnow()
            
            # Parse PR URL
            repo, pr_number = self.github_client.parse_pr_url(pr_url)
            logger.debug(f"Parsed PR URL: {repo}#{pr_number}")
            
            # Fetch PR data
            pr_data = await self.github_client.fetch_pr_data(repo, pr_number)
            
            # Use the metadata from pr_data
            pr_metadata = pr_data.metadata
            
            # Get diff content from pr_data
            diff_content = pr_data.diff_content
            
            logger.info(f"Diff content length: {len(diff_content)} characters")
            logger.debug(f"First 500 chars of diff: {diff_content[:500]}")
            
            # Parse diff
            parsed_diff = self.diff_parser.parse(diff_content)
            
            logger.info(f"Parsed diff: {len(parsed_diff.files)} files found")
            if parsed_diff.files:
                for f in parsed_diff.files[:5]:  # Log first 5 files
                    logger.info(f"  File: {f.file_path}, additions: {len(f.additions)}, deletions: {len(f.deletions)}")
            else:
                logger.warning("No files found in parsed diff!")
            
            # Perform review
            response = await self._perform_review(
                parsed_diff=parsed_diff,
                pr_metadata=pr_metadata,
                config=config
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Review completed in {execution_time:.2f}s")
            
            return response
            
        except GitHubAPIError as e:
            logger.error(f"GitHub API error: {e}")
            raise
        except DiffParseError as e:
            logger.error(f"Diff parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Review failed: {e}", exc_info=True)
            raise
    
    async def review_from_diff(
        self,
        diff_content: str,
        repository: Optional[str] = None,
        pr_number: Optional[int] = None,
        config: Optional[ReviewConfig] = None
    ) -> ReviewResponse:
        """
        Review code from manual diff input
        
        Args:
            diff_content: Git diff content
            repository: Optional repository name
            pr_number: Optional PR number
            config: Review configuration
            
        Returns:
            ReviewResponse with findings and formatted comments
        """
        try:
            logger.info("Starting review from manual diff")
            start_time = datetime.utcnow()
            
            # Parse diff
            parsed_diff = self.diff_parser.parse(diff_content)
            
            # Create minimal PR metadata (always create it, never None)
            pr_metadata = PRMetadata(
                repository=repository or "manual-review",
                pr_number=pr_number or 0,
                title="Manual Review",
                author="",
                commit_sha="",
                base_branch="",
                head_branch=""
            )
            
            # Perform review
            response = await self._perform_review(
                parsed_diff=parsed_diff,
                pr_metadata=pr_metadata,
                config=config
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Review completed in {execution_time:.2f}s")
            
            return response
            
        except DiffParseError as e:
            logger.error(f"Diff parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Review failed: {e}", exc_info=True)
            raise
    
    async def _perform_review(
        self,
        parsed_diff: ParsedDiff,
        pr_metadata: Optional[PRMetadata] = None,
        config: Optional[ReviewConfig] = None
    ) -> ReviewResponse:
        """
        Perform the actual review workflow
        
        Args:
            parsed_diff: Parsed diff with file changes
            pr_metadata: Optional PR metadata
            config: Review configuration
            
        Returns:
            ReviewResponse with findings and formatted comments
        """
        start_time = datetime.utcnow()

        # Use default config if not provided
        if config is None:
            config = ReviewConfig()
        
        # Create orchestrator with config
        orchestrator = ReviewOrchestrator(self.llm, config)
        
        # Orchestrate review across all agents
        logger.info("Orchestrating multi-agent review")
        findings = await orchestrator.orchestrate_review(parsed_diff, pr_metadata)
        
        logger.info(f"Review complete: {len(findings)} findings")
        
        # Generate formatted comments
        logger.info("Generating formatted comments")
        formatted_review = self.comment_generator.generate_comments(findings)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Persist review if database session available
        review_id = None
        if self.db_session and pr_metadata:
            try:
                logger.info("Persisting review to database")
                repository = ReviewRepository(self.db_session)
                review_id = await repository.save_review(
                    pr_metadata=pr_metadata,
                    commit_sha=pr_metadata.commit_sha or "",
                    findings=findings,
                    config=config,
                    summary=formatted_review.summary,
                    execution_time=execution_time
                )
                logger.info(f"Review persisted with ID: {review_id}")
            except Exception as e:
                logger.error(f"Failed to persist review: {e}", exc_info=True)
                if hasattr(self.db_session, "rollback"):
                    await self.db_session.rollback()
                # Continue even if persistence fails
        
        # Create response
        response = ReviewResponse(
            review_id=review_id or "",
            pr_metadata=pr_metadata,
            findings=findings,
            summary=formatted_review.summary,
            formatted_comments=formatted_review.markdown_output,
            timestamp=datetime.utcnow()
        )
        
        return response
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the service"""
        return {
            "github_client_available": self.github_client is not None,
            "database_available": self.db_session is not None,
            "components": {
                "diff_parser": "DiffParser",
                "comment_generator": "CommentGenerator",
                "orchestrator": "ReviewOrchestrator"
            }
        }
