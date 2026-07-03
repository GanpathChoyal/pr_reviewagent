"""
GitHub API Client for fetching PR data and diffs
"""
import re
import asyncio
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
from github import Github, GithubException
from github.PullRequest import PullRequest

from models import PRData, PRMetadata
from config import settings

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors"""
    pass


class GitHubAPIClient:
    """Client for interacting with GitHub API"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub API client
        
        Args:
            token: GitHub personal access token. If None, uses token from settings.
        """
        self.token = token or settings.github_token
        self._github = None
        self._http_client = None
        
    def _get_github_client(self) -> Github:
        """Get authenticated GitHub client"""
        if self._github is None:
            if not self.token:
                raise GitHubAPIError(
                    "GitHub token is required. Set GITHUB_TOKEN environment variable "
                    "or provide token parameter."
                )
            self._github = Github(self.token)
        return self._github
    
    def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for direct API calls"""
        if self._http_client is None:
            headers = {}
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            headers["Accept"] = "application/vnd.github.v3+json"
            
            self._http_client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
                follow_redirects=True
            )
        return self._http_client
    
    def parse_pr_url(self, url: str) -> Tuple[str, int]:
        """
        Parse GitHub PR URL to extract repository and PR number
        
        Args:
            url: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)
            
        Returns:
            Tuple of (repository, pr_number)
            
        Raises:
            GitHubAPIError: If URL format is invalid
        """
        # Support various GitHub URL formats
        patterns = [
            r"github\.com/([^/]+/[^/]+)/pull/(\d+)",
            r"github\.com/([^/]+/[^/]+)/pulls/(\d+)",
            r"api\.github\.com/repos/([^/]+/[^/]+)/pulls/(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                repository = match.group(1)
                pr_number = int(match.group(2))
                return repository, pr_number
        
        raise GitHubAPIError(
            f"Invalid GitHub PR URL format: {url}. "
            "Expected format: https://github.com/owner/repo/pull/123"
        )
    
    async def fetch_pr_data(self, repo: str, pr_number: int) -> PRData:
        """
        Fetch complete PR data including metadata and diff
        
        Args:
            repo: Repository in format "owner/repo"
            pr_number: Pull request number
            
        Returns:
            PRData object with metadata and diff content
            
        Raises:
            GitHubAPIError: If API request fails
        """
        try:
            # Fetch PR metadata
            metadata = await self._fetch_pr_metadata(repo, pr_number)
            
            # Fetch PR diff
            diff_content = await self.fetch_pr_diff(repo, pr_number)
            
            # Get list of changed files
            files_changed = await self._fetch_changed_files(repo, pr_number)
            
            return PRData(
                metadata=metadata,
                diff_content=diff_content,
                files_changed=files_changed
            )
            
        except GithubException as e:
            error_msg = self._format_github_error(e)
            logger.error(f"GitHub API error fetching PR {repo}#{pr_number}: {error_msg}")
            raise GitHubAPIError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching PR data: {str(e)}"
            logger.error(f"Error fetching PR {repo}#{pr_number}: {error_msg}")
            raise GitHubAPIError(error_msg)
    
    async def _fetch_pr_metadata(self, repo: str, pr_number: int) -> PRMetadata:
        """Fetch PR metadata using PyGithub"""
        github = self._get_github_client()
        
        # Use asyncio to run the synchronous GitHub API call
        def _get_pr_sync():
            repository = github.get_repo(repo)
            return repository.get_pull(pr_number)
        
        pr: PullRequest = await asyncio.get_event_loop().run_in_executor(
            None, _get_pr_sync
        )
        
        return PRMetadata(
            repository=repo,
            pr_number=pr_number,
            title=pr.title,
            author=pr.user.login,
            commit_sha=pr.head.sha,
            base_branch=pr.base.ref,
            head_branch=pr.head.ref
        )
    
    async def fetch_pr_diff(self, repo: str, pr_number: int) -> str:
        """
        Fetch PR diff content
        
        Args:
            repo: Repository in format "owner/repo"
            pr_number: Pull request number
            
        Returns:
            Raw diff content as string
            
        Raises:
            GitHubAPIError: If API request fails
        """
        client = self._get_http_client()
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        
        try:
            # Request diff format
            headers = {"Accept": "application/vnd.github.v3.diff"}
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            return response.text
            
        except httpx.HTTPStatusError as e:
            error_msg = self._format_http_error(e)
            logger.error(f"HTTP error fetching diff for {repo}#{pr_number}: {error_msg}")
            raise GitHubAPIError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error fetching diff: {str(e)}"
            logger.error(f"Error fetching diff for {repo}#{pr_number}: {error_msg}")
            raise GitHubAPIError(error_msg)
    
    async def _fetch_changed_files(self, repo: str, pr_number: int) -> list[str]:
        """Fetch list of files changed in the PR"""
        client = self._get_http_client()
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            files_data = response.json()
            return [file_info["filename"] for file_info in files_data]
            
        except httpx.HTTPStatusError as e:
            # Log warning but don't fail - this is supplementary data
            logger.warning(f"Could not fetch changed files for {repo}#{pr_number}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error fetching changed files for {repo}#{pr_number}: {e}")
            return []
    
    def _format_github_error(self, error: GithubException) -> str:
        """Format GitHub API error into user-friendly message"""
        if error.status == 401:
            return "Authentication failed. Please check your GitHub token."
        elif error.status == 403:
            if "rate limit" in str(error).lower():
                return "GitHub API rate limit exceeded. Please try again later."
            else:
                return "Access forbidden. Check repository permissions and token scope."
        elif error.status == 404:
            return "Repository or pull request not found. Check the URL and permissions."
        elif error.status == 422:
            return "Invalid request. The pull request may be in an invalid state."
        else:
            return f"GitHub API error (status {error.status}): {error.data.get('message', str(error))}"
    
    def _format_http_error(self, error: httpx.HTTPStatusError) -> str:
        """Format HTTP error into user-friendly message"""
        if error.response.status_code == 401:
            return "Authentication failed. Please check your GitHub token."
        elif error.response.status_code == 403:
            return "Access forbidden or rate limit exceeded."
        elif error.response.status_code == 404:
            return "Repository or pull request not found."
        else:
            try:
                error_data = error.response.json()
                message = error_data.get("message", "Unknown error")
            except:
                message = error.response.text or "Unknown error"
            
            return f"HTTP error (status {error.response.status_code}): {message}"
    
    async def validate_token(self) -> bool:
        """
        Validate GitHub token by making a simple API call
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self.token:
            return False
        
        try:
            client = self._get_http_client()
            response = await client.get("https://api.github.com/user")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close HTTP client connections"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Retry decorator for API calls
def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry API calls on failure"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"API call failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {max_retries} attempts: {e}")
                        raise GitHubAPIError(f"API call failed after {max_retries} attempts: {e}")
                except Exception as e:
                    # Don't retry on non-network errors
                    raise e
            
            # This shouldn't be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


# Apply retry logic to key methods
GitHubAPIClient.fetch_pr_diff = retry_on_failure(max_retries=3)(GitHubAPIClient.fetch_pr_diff)
GitHubAPIClient._fetch_changed_files = retry_on_failure(max_retries=2)(GitHubAPIClient._fetch_changed_files)