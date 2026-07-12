"""
Abstract base class for analyzer agents
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel

from models import Finding, ReviewContext, SeverityLevel, AnalysisCategory

logger = logging.getLogger(__name__)


class AgentConfig:
    """Configuration for analyzer agents"""
    
    def __init__(
        self,
        max_tokens: int = 4000,  # Increased for more detailed analysis
        temperature: float = 0.1,
        timeout: int = 300,  # 5 minutes per agent
        max_retries: int = 2,
        custom_rules: Optional[Dict[str, Any]] = None
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.custom_rules = custom_rules or {}


class AnalyzerAgent(ABC):
    """Abstract base class for code analyzer agents"""
    
    def __init__(self, llm: BaseLanguageModel, config: Optional[AgentConfig] = None):
        """
        Initialize analyzer agent
        
        Args:
            llm: Language model for analysis
            config: Agent configuration
        """
        self.llm = llm
        self.config = config or AgentConfig()
        self.agent_name = self.__class__.__name__.lower().replace('agent', '')
        
    @abstractmethod
    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """
        Analyze code changes and return findings
        
        Args:
            context: Review context with file changes and configuration
            
        Returns:
            List of findings from the analysis
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this analyzer
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def get_analysis_category(self) -> AnalysisCategory:
        """
        Get the analysis category for this agent
        
        Returns:
            Analysis category enum value
        """
        pass
    
    def create_prompt(self, context: ReviewContext) -> List[BaseMessage]:
        """
        Create prompt messages for LLM analysis
        
        Args:
            context: Review context with file changes
            
        Returns:
            List of messages for the LLM
        """
        # System message with agent-specific instructions
        system_prompt = self.get_system_prompt()
        
        # Add custom rules if available
        if self.config.custom_rules:
            custom_rules_text = self._format_custom_rules(self.config.custom_rules)
            system_prompt += f"\n\nCustom Rules:\n{custom_rules_text}"
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Human message with code changes
        human_prompt = self._create_human_prompt(context)
        messages.append(HumanMessage(content=human_prompt))
        
        return messages
    
    def _create_human_prompt(self, context: ReviewContext) -> str:
        """Create human prompt with code changes"""
        prompt_parts = []
        
        # Add context information
        if context.pr_metadata:
            prompt_parts.append(f"Pull Request: {context.pr_metadata.title}")
            prompt_parts.append(f"Repository: {context.pr_metadata.repository}")
            prompt_parts.append(f"Author: {context.pr_metadata.author}")
            prompt_parts.append("")
        
        # Add file changes
        prompt_parts.append("Code Changes to Analyze:")
        prompt_parts.append("=" * 40)
        
        for file_change in context.file_changes:
            if file_change.is_binary:
                prompt_parts.append(f"\nFile: {file_change.file_path} (binary file - skipped)")
                continue
            
            prompt_parts.append(f"\nFile: {file_change.file_path}")
            prompt_parts.append(f"Language: {file_change.language}")
            
            # Add additions
            if file_change.additions:
                prompt_parts.append("\nAdditions:")
                for line_change in file_change.additions:
                    prompt_parts.append(f"+{line_change.line_number}: {line_change.content}")
            
            # Add deletions
            if file_change.deletions:
                prompt_parts.append("\nDeletions:")
                for line_change in file_change.deletions:
                    prompt_parts.append(f"-{line_change.line_number}: {line_change.content}")
            
            # Add modifications
            if file_change.modifications:
                prompt_parts.append("\nModifications:")
                for line_change in file_change.modifications:
                    prompt_parts.append(f"~{line_change.line_number}: {line_change.content}")
            
            prompt_parts.append("-" * 40)
        
        # Add analysis instructions
        prompt_parts.append("\nPlease analyze the above code changes and return your findings in JSON format.")
        prompt_parts.append("Focus on issues relevant to your specialization.")
        prompt_parts.append("Return an empty array if no issues are found.")
        
        return "\n".join(prompt_parts)
    
    def _format_custom_rules(self, custom_rules: Dict[str, Any]) -> str:
        """Format custom rules for inclusion in prompt"""
        rules_text = []
        for key, value in custom_rules.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, indent=2)
            rules_text.append(f"- {key}: {value}")
        return "\n".join(rules_text)
    
    async def parse_llm_response(self, response: str) -> List[Finding]:
        """
        Parse LLM response into Finding objects
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of Finding objects
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON
            findings_data = json.loads(cleaned_response)
            
            # Ensure it's a list
            if not isinstance(findings_data, list):
                logger.warning(f"Expected list in LLM response, got {type(findings_data)}")
                return []
            
            # Convert to Finding objects
            findings = []
            for item in findings_data:
                try:
                    finding = self._create_finding_from_dict(item)
                    if finding:
                        findings.append(finding)
                except Exception as e:
                    logger.warning(f"Failed to create finding from {item}: {e}")
                    continue
            
            return findings
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            # Return empty list instead of raising - be resilient to LLM errors
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract JSON content"""
        # Remove markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # If response doesn't start with [ or {, try to find JSON content
        if not response.startswith(('[', '{')):
            # Look for JSON array or object in the response
            for char in ['[', '{']:
                start_idx = response.find(char)
                if start_idx != -1:
                    response = response[start_idx:]
                    break
        
        return response
    
    def _create_finding_from_dict(self, data: Dict[str, Any]) -> Optional[Finding]:
        """Create Finding object from dictionary data"""
        try:
            # Validate required fields
            required_fields = ['file_path', 'line_number', 'description']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field '{field}' in finding data")
                    return None
            
            # Parse severity
            severity_str = data.get('severity', 'medium').lower()
            try:
                severity = SeverityLevel(severity_str)
            except ValueError:
                logger.warning(f"Invalid severity '{severity_str}', defaulting to medium")
                severity = SeverityLevel.MEDIUM
            
            # Create finding
            finding = Finding(
                file_path=str(data['file_path']),
                line_number=int(data['line_number']),
                severity=severity,
                category=self.get_analysis_category(),
                description=str(data['description']),
                suggestion=data.get('suggestion'),
                agent_source=self.agent_name
            )
            
            return finding
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to create finding from data {data}: {e}")
            return None
    
    async def _invoke_llm_with_retry(self, messages: List[BaseMessage]) -> str:
        """
        Invoke LLM with retry logic
        
        Args:
            messages: Messages to send to LLM
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        logger.info(f"[{self.agent_name}] Invoking LLM with {len(messages)} messages")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Configure LLM parameters
                llm_kwargs = {
                    'max_tokens': self.config.max_tokens,
                    'temperature': self.config.temperature
                }
                
                logger.debug(f"[{self.agent_name}] LLM call attempt {attempt + 1} with params: {llm_kwargs}")
                
                # Invoke LLM
                response = await self.llm.ainvoke(messages, **llm_kwargs)
                
                logger.info(f"[{self.agent_name}] LLM responded successfully")
                
                # Extract content from response
                if hasattr(response, 'content'):
                    content = response.content
                    logger.debug(f"[{self.agent_name}] Response length: {len(content)} characters")
                    return content
                else:
                    content = str(response)
                    logger.debug(f"[{self.agent_name}] Response (as string) length: {len(content)} characters")
                    return content
                    
            except Exception as e:
                last_exception = e
                logger.error(f"[{self.agent_name}] LLM invocation error: {type(e).__name__}: {str(e)}")
                if attempt < self.config.max_retries:
                    logger.warning(f"[{self.agent_name}] Retrying (attempt {attempt + 1}/{self.config.max_retries})...")
                    # Simple backoff - wait 1 second between retries
                    import asyncio
                    await asyncio.sleep(1.0)
                else:
                    logger.error(f"[{self.agent_name}] All retry attempts exhausted")
        
        # If we get here, all retries failed
        logger.error(f"[{self.agent_name}] LLM invocation completely failed")
        raise last_exception or Exception("LLM invocation failed")
    
    def get_expected_output_format(self) -> str:
        """Get description of expected JSON output format"""
        return """
Expected JSON format:
[
  {
    "file_path": "path/to/file.py",
    "line_number": 42,
    "severity": "high|medium|low|critical",
    "description": "Clear description of the issue",
    "suggestion": "Optional suggestion for fixing the issue"
  }
]

Return an empty array [] if no issues are found.
"""
    
    def validate_context(self, context: ReviewContext) -> bool:
        """
        Validate that the context is suitable for analysis
        
        Args:
            context: Review context to validate
            
        Returns:
            True if context is valid for analysis
        """
        if not context.file_changes:
            logger.debug("No file changes to analyze")
            return False
        
        # Check if there are any non-binary files to analyze
        text_files = [f for f in context.file_changes if not f.is_binary]
        if not text_files:
            logger.debug("No text files to analyze (all binary)")
            return False
        
        # Check if any files have actual changes
        has_changes = any(
            f.additions or f.deletions or f.modifications 
            for f in text_files
        )
        
        if not has_changes:
            logger.debug("No actual code changes to analyze")
            return False
        
        return True
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            'name': self.agent_name,
            'category': self.get_analysis_category().value,
            'config': {
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'timeout': self.config.timeout,
                'max_retries': self.config.max_retries
            }
        }