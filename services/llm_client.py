"""
LLM Client wrapper for NVIDIA NIM integration
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Union, List
from enum import Enum
import httpx

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from config import settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    NVIDIA = "nvidia"
    GROQ = "groq"


class LLMClientError(Exception):
    """Custom exception for LLM client errors"""
    pass


class NimLlmClient:
    """NVIDIA NIM LLM Client - Wrapper for NVIDIA NIM API with per-agent model selection"""
    
    # Model mapping for different analyzers
    AGENT_MODELS = {
    "logic_analyzer": "openai/gpt-oss-120b",
    "readability_analyzer": "openai/gpt-oss-120b",
    "performance_analyzer": "openai/gpt-oss-120b",
    "security_analyzer": "openai/gpt-oss-120b",
    "default": "openai/gpt-oss-120b",
}
    
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        agent_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs
    ):
        """
        Initialize NVIDIA NIM client
        
        Args:
            api_key: NVIDIA API key
            model: Specific model to use (overrides agent-based selection)
            agent_name: Name of the agent (for automatic model selection)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key
        
        # Select model based on agent name or use provided model
        if model:
            self.model = model
        elif agent_name and agent_name in self.AGENT_MODELS:
            self.model = self.AGENT_MODELS[agent_name]
        else:
            self.model = self.AGENT_MODELS["default"]
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
        self._http_client = None
        
        logger.info(f"NimLlmClient initialized with model: {self.model}")
    
    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self._http_client = httpx.AsyncClient(
                headers=headers,
                timeout=60.0
            )
        return self._http_client
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to NVIDIA NIM format"""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            else:
                # Default to user role for other message types
                formatted_messages.append({"role": "user", "content": str(msg.content)})
        return formatted_messages
    
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Async invoke the NVIDIA NIM API"""
        client = self._get_http_client()
        
        # Convert messages to NVIDIA format
        formatted_messages = self._messages_to_prompt(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        try:
            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            message = result["choices"][0]["message"]
            
            # Handle thinking models that use reasoning_content
            content = message.get("content")
            if content is None:
                # For thinking models, use reasoning_content
                content = message.get("reasoning_content", "")
            
            # Return a simple object with content attribute for compatibility
            class Response:
                def __init__(self, content):
                    self.content = content
            
            return Response(content)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"NVIDIA NIM API error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"NVIDIA NIM request failed: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Sync invoke (runs async version)"""
        return asyncio.run(self.ainvoke(messages, **kwargs))
    
    async def aclose(self):
        """Close HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class GROQLlmClient:
    """Groq (GroqCloud) API client with per-agent API key selection.

    Groq exposes an OpenAI-compatible /chat/completions endpoint, so the
    request/response shape mirrors NimLlmClient. This is Groq
    (api.groq.com, gsk_... keys, fast LPU inference of open models),
    NOT xAI's Grok (api.x.ai) — different company, similar name.
    """

    AGENT_MODELS = {
    "logic_analyzer": "openai/gpt-oss-120b",
    "readability_analyzer": "openai/gpt-oss-120b",
    "performance_analyzer": "openai/gpt-oss-120b",
    "security_analyzer": "openai/gpt-oss-120b",
    "default": "openai/gpt-oss-120b",
}

    AGENT_KEY_SETTINGS = {
        "logic_analyzer": "GROQ_logic_api_key",
        "readability_analyzer": "GROQ_readability_api_key",
        "performance_analyzer": "GROQ_performance_api_key",
        "security_analyzer": "GROQ_security_api_key",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        agent_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.agent_name = agent_name
        self.api_key = api_key or self._api_key_for_agent(agent_name)

        if not self.api_key and not self._has_agent_api_keys():
            raise LLMClientError(
                "Groq API key is required. Set GROQ_API_KEY or the per-agent "
                "keys: GROQ_LOGIC_API_KEY, GROQ_READABILITY_API_KEY, "
                "GROQ_PERFORMANCE_API_KEY, GROQ_SECURITY_API_KEY."
            )

        if model:
            self.model = model
        elif agent_name and agent_name in self.AGENT_MODELS:
            self.model = self.AGENT_MODELS[agent_name]
        else:
            self.model = settings.llm_model or self.AGENT_MODELS["default"]

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self._http_client = None

        logger.info(
            "GROQLlmClient initialized with model: %s%s",
            self.model,
            f" for {agent_name}" if agent_name else "",
        )

    def for_agent(self, agent_name: str) -> "GROQLlmClient":
        """Create a GROQ client using this agent's configured API key."""
        return GROQLlmClient(
            api_key=self._api_key_for_agent(agent_name),
            model=self.AGENT_MODELS.get(agent_name, self.model),
            agent_name=agent_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _api_key_for_agent(self, agent_name: Optional[str]) -> str:
        if agent_name:
            setting_name = self.AGENT_KEY_SETTINGS.get(agent_name)
            agent_key = getattr(settings, setting_name, "") if setting_name else ""
            if agent_key:
                return agent_key

        return getattr(settings, "GROQ_api_key", "")

    def _has_agent_api_keys(self) -> bool:
        return any(
            getattr(settings, setting_name, "")
            for setting_name in self.AGENT_KEY_SETTINGS.values()
        )

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._http_client = httpx.AsyncClient(headers=headers, timeout=60.0)
        return self._http_client

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI-style chat format."""
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            else:
                formatted_messages.append({"role": "user", "content": str(msg.content)})
        return formatted_messages

    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Async invoke the Groq API."""
        client = self._get_http_client()
        formatted_messages = self._messages_to_prompt(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        try:
            if not self.api_key:
                raise LLMClientError(
                    f"No Groq API key configured for {self.agent_name or 'this client'}"
                )

            response = await client.post(self.endpoint, json=payload)
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]
            content = message.get("content", "") or ""

            class Response:
                def __init__(self, content):
                    self.content = content

            return Response(content)

        except httpx.HTTPStatusError as e:
            error_msg = f"Groq API error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"Groq request failed: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)

    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """Sync invoke (runs async version)."""
        return asyncio.run(self.ainvoke(messages, **kwargs))

    async def aclose(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class LLMClient:
    """Wrapper for LLM providers with unified interface"""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.NVIDIA,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider to use
            model: Model name (uses default if not specified)
            api_key: API key (uses settings if not specified)
            **kwargs: Additional model parameters
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._client = None
        
    def _get_client(self) -> BaseLanguageModel:
        """Get or create LLM client instance"""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def _create_client(self) -> BaseLanguageModel:
        """Create LLM client based on provider"""
        if self.provider == LLMProvider.NVIDIA:
            return self._create_nvidia_client()
        elif self.provider == LLMProvider.GROQ:
            return self._create_GROQ_client()
        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")
    
    def _create_nvidia_client(self) -> NimLlmClient:
        """Create NVIDIA NIM client"""
        api_key = self.api_key or settings.nvidia_api_key
        if not api_key:
            raise LLMClientError(
                "NVIDIA API key is required. Set NVIDIA_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        # Default model and parameters
        # Use model from settings if available, otherwise use default
        default_model = settings.llm_model if settings.llm_provider.lower() == "nvidia" else None
        model = self.model or default_model
        
        default_params = {
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        
        # Merge with user-provided parameters
        params = {**default_params, **self.kwargs}
        
        # Extract agent_name if provided in kwargs
        agent_name = params.pop("agent_name", None)
        
        return NimLlmClient(
            api_key=api_key,
            model=model,
            agent_name=agent_name,
            **params
        )

    def _create_GROQ_client(self) -> GROQLlmClient:
        """Create Groq client."""
        default_model = settings.llm_model if settings.llm_provider.lower() == "groq" else None
        model = self.model or default_model

        default_params = {
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        params = {**default_params, **self.kwargs}
        agent_name = params.pop("agent_name", None)

        return GROQLlmClient(
            api_key=self.api_key,
            model=model,
            agent_name=agent_name,
            **params
        )
    
    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> str:
        """
        Async invoke LLM with messages
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters for this invocation
            
        Returns:
            LLM response content as string
            
        Raises:
            LLMClientError: If invocation fails
        """
        try:
            client = self._get_client()
            
            # Invoke with timeout
            response = await asyncio.wait_for(
                client.ainvoke(messages, **kwargs),
                timeout=kwargs.get('timeout', 30)
            )
            
            # Extract content
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except asyncio.TimeoutError:
            error_msg = "LLM request timed out"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"LLM invocation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    def invoke(self, messages: list[BaseMessage], **kwargs) -> str:
        """
        Sync invoke LLM with messages
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters for this invocation
            
        Returns:
            LLM response content as string
        """
        try:
            client = self._get_client()
            response = client.invoke(messages, **kwargs)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            error_msg = f"LLM invocation failed: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def test_connection(self) -> bool:
        """
        Test LLM connection with a simple request
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            from langchain.schema import HumanMessage
            
            test_messages = [
                HumanMessage(content="Hello, please respond with 'OK' if you can hear me.")
            ]
            
            response = await self.ainvoke(test_messages, max_tokens=10)
            return "ok" in response.lower()
            
        except Exception as e:
            logger.warning(f"LLM connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.provider.value,
            "model": self.model or self._get_default_model(),
            "has_api_key": bool(self.api_key or self._get_api_key_from_settings()),
        }
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        defaults = {
            LLMProvider.NVIDIA: "meta/llama-3.1-8b-instruct",
            LLMProvider.GROQ: "llama-3.3-70b-versatile",
        }
        return defaults.get(self.provider, "unknown")
    
    def _get_api_key_from_settings(self) -> Optional[str]:
        """Get API key from settings based on provider"""
        if self.provider == LLMProvider.NVIDIA:
            return settings.nvidia_api_key
        if self.provider == LLMProvider.GROQ:
            return (
                getattr(settings, "GROQ_api_key", "")
                or getattr(settings, "GROQ_logic_api_key", "")
                or getattr(settings, "GROQ_readability_api_key", "")
                or getattr(settings, "GROQ_performance_api_key", "")
                or getattr(settings, "GROQ_security_api_key", "")
            )
        return None


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(
        provider: Union[str, LLMProvider] = LLMProvider.NVIDIA,
        **kwargs
    ) -> LLMClient:
        """
        Create LLM client
        
        Args:
            provider: Provider name or enum
            **kwargs: Additional parameters
            
        Returns:
            LLM client instance
        """
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                raise LLMClientError(f"Unknown provider: {provider}")
        
        return LLMClient(provider=provider, **kwargs)
    
    @staticmethod
    def create_nvidia_client(model: str = "meta/llama-3.1-8b-instruct", **kwargs) -> LLMClient:
        """Create NVIDIA NIM client with specific model"""
        return LLMClient(
            provider=LLMProvider.NVIDIA,
            model=model,
            **kwargs
        )

    @staticmethod
    def create_GROQ_client(model: str = "llama-3.3-70b-versatile", **kwargs) -> LLMClient:
        """Create Groq client with specific model."""
        return LLMClient(
            provider=LLMProvider.GROQ,
            model=model,
            **kwargs
        )
    
    @staticmethod
    def create_default_client() -> LLMClient:
        """Create default client based on available API keys"""
        has_GROQ_key = any([
            getattr(settings, "GROQ_api_key", ""),
            getattr(settings, "GROQ_logic_api_key", ""),
            getattr(settings, "GROQ_readability_api_key", ""),
            getattr(settings, "GROQ_performance_api_key", ""),
            getattr(settings, "GROQ_security_api_key", ""),
        ])

        if settings.llm_provider.lower() == "groq" and has_GROQ_key:
            return LLMClientFactory.create_GROQ_client(model=settings.llm_model)

        # Try NVIDIA NIM
        if settings.nvidia_api_key:
            return LLMClientFactory.create_nvidia_client()
        
        # No API keys available
        raise LLMClientError(
            "No LLM API keys found. Set GROQ_API_KEY or NVIDIA_API_KEY environment variable."
        )


# Retry wrapper for LLM operations
async def with_retry(
    operation,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Execute operation with exponential backoff retry
    
    Args:
        operation: Async operation to execute
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        
    Returns:
        Operation result
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {current_delay}s: {e}"
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
    
    raise last_exception or Exception("Operation failed after retries")


# Global client instance for convenience
_default_client: Optional[LLMClient] = None


def get_default_llm_client() -> LLMClient:
    """Get or create default LLM client"""
    global _default_client
    
    if _default_client is None:
        _default_client = LLMClientFactory.create_default_client()
    
    return _default_client


def set_default_llm_client(client: LLMClient):
    """Set default LLM client"""
    global _default_client
    _default_client = client
