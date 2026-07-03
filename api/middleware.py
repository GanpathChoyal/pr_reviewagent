"""
API middleware for security and rate limiting
"""
import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, requests_per_minute: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Get client IP
        client_ip = request.client.host
        
        # Clean up old entries periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
            self.last_cleanup = current_time
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
            )
        
        # Record request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago
        
        # Get recent requests
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > cutoff_time
        ]
        
        # Update stored requests
        self.requests[client_ip] = recent_requests
        
        # Check limit
        return len(recent_requests) < self.requests_per_minute
    
    def _cleanup_old_requests(self):
        """Remove old request records"""
        current_time = time.time()
        cutoff_time = current_time - 120  # Keep last 2 minutes
        
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[client_ip]:
                del self.requests[client_ip]


async def verify_api_key(api_key: Optional[str] = None) -> bool:
    """
    Verify API key (optional authentication)
    
    Args:
        api_key: API key from header
        
    Returns:
        True if valid or no key required, False otherwise
    """
    # If no API key is configured, allow all requests
    if not hasattr(settings, 'api_key') or not settings.api_key:
        return True
    
    # If API key is configured, verify it
    if not api_key:
        return False
    
    return api_key == settings.api_key


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response