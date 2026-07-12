"""
FastAPI endpoints for PR reviews
"""
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from models.api_models import ReviewRequest, ReviewResponse
from models import ReviewConfig, SeverityLevel, AnalysisCategory
from services import PRReviewService, LLMClientFactory, LLMProvider
from repositories import get_db, ReviewRepository, HistoryFilters
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reviews", tags=["reviews"])


def get_llm():
    """Get LLM instance"""
    try:
        llm_factory = LLMClientFactory()
        
        # Get the appropriate API key based on provider
        provider = settings.llm_provider.lower()
        if provider == "nvidia":
            api_key = settings.nvidia_api_key
        elif provider == "groq":
            api_key = settings.GROQ_api_key 
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Create LLM client wrapper
        llm_client = llm_factory.create_client(
            provider=LLMProvider(provider),
            api_key=api_key,
            model=settings.llm_model
        )
        
        # Return the actual LangChain model, not the wrapper
        # The agents expect a BaseLanguageModel, not LLMClient
        return llm_client._get_client()
        
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize LLM client"
        )


@router.post("", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED)
async def create_review(
    request: ReviewRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new code review
    
    Accepts either a GitHub PR URL or manual diff content.
    Returns structured review findings with formatted comments.
    """
    try:
        logger.info(f"Received review request: {request.pr_url or 'manual diff'}")
        
        # Get LLM client
        llm = get_llm()
        
        # Create review service
        service = PRReviewService(
            llm=llm,
            github_token=request.github_token or settings.github_token,
            db_session=db if settings.persist_reviews else None
        )
        
        # Create config from request
        config = ReviewConfig(
            severity_threshold=request.config.severity_threshold if request.config else None,
            enabled_categories=request.config.enabled_categories if request.config else [
                AnalysisCategory.LOGIC.value,
                AnalysisCategory.READABILITY.value,
                AnalysisCategory.PERFORMANCE.value,
                AnalysisCategory.SECURITY.value
            ],
            custom_rules=request.config.custom_rules if request.config else None
        )
        
        # Perform review based on input type
        if request.pr_url:
            response = await service.review_pr_from_url(
                pr_url=request.pr_url,
                config=config
            )
        elif request.diff_content:
            response = await service.review_from_diff(
                diff_content=request.diff_content,
                repository=request.repository,
                pr_number=request.pr_number,
                config=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either pr_url or diff_content must be provided"
            )
        
        logger.info(f"Review completed: {len(response.findings)} findings")
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Review failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Review processing failed: {str(e)}"
        )


@router.get("/history", response_model=List[ReviewResponse])
async def get_review_history(
    repository: Optional[str] = Query(None, description="Filter by repository"),
    pr_number: Optional[int] = Query(None, description="Filter by PR number"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get review history with optional filtering
    
    Supports filtering by:
    - Repository name
    - PR number
    - Date range
    - Severity level
    - Category
    """
    try:
        logger.info(f"Fetching review history: repo={repository}, pr={pr_number}")
        
        # Create repository instance
        repo = ReviewRepository(db)
        
        # Create filters
        filters = HistoryFilters(
            repository=repository,
            pr_number=pr_number,
            start_date=start_date,
            end_date=end_date,
            severity=severity,
            category=category,
            limit=limit,
            offset=offset
        )
        
        # Query reviews
        review_results = await repo.query_reviews(filters)
        
        # Convert to response models
        responses = []
        for result in review_results:
            response = ReviewResponse(
                review_id=result.review_id,
                pr_metadata=result.pr_metadata,
                findings=result.findings,
                summary=None,  # Summary would need to be reconstructed
                formatted_comments="",  # Not stored in DB
                timestamp=result.timestamp
            )
            responses.append(response)
        
        logger.info(f"Returned {len(responses)} review results")
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to fetch review history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch review history: {str(e)}"
        )


@router.get("/{review_id}/status")
async def get_review_status(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of a specific review
    
    Returns review status and basic information.
    """
    try:
        logger.info(f"Fetching status for review: {review_id}")
        
        # Create repository instance
        repo = ReviewRepository(db)
        
        # Get review
        review_result = await repo.get_review(review_id)
        
        if not review_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review {review_id} not found"
            )
        
        # Return status information
        return {
            "review_id": review_result.review_id,
            "status": "completed",  # All stored reviews are completed
            "repository": review_result.pr_metadata.repository,
            "pr_number": review_result.pr_metadata.pr_number,
            "timestamp": review_result.timestamp,
            "findings_count": len(review_result.findings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch review status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch review status: {str(e)}"
        )


@router.get("/{review_id}", response_model=ReviewResponse)
async def get_review(
    review_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific review by ID
    
    Returns complete review details including all findings.
    """
    try:
        logger.info(f"Fetching review: {review_id}")
        
        # Create repository instance
        repo = ReviewRepository(db)
        
        # Get review
        review_result = await repo.get_review(review_id)
        
        if not review_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review {review_id} not found"
            )
        
        # Convert to response model
        response = ReviewResponse(
            review_id=review_result.review_id,
            pr_metadata=review_result.pr_metadata,
            findings=review_result.findings,
            summary=None,  # Would need to be reconstructed
            formatted_comments="",  # Not stored in DB
            timestamp=review_result.timestamp
        )
        
        logger.info(f"Retrieved review {review_id} with {len(response.findings)} findings")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch review: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch review: {str(e)}"
        )
