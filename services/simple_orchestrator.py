"""
Simple Review Orchestrator - Direct agent execution without LangGraph
"""
import logging
import asyncio
from typing import List, Optional, Any
from datetime import datetime

from langchain_core.language_models import BaseLanguageModel

from agents import (
    LogicAnalyzerAgent,
    ReadabilityAnalyzerAgent,
    PerformanceAnalyzerAgent,
    SecurityAnalyzerAgent,
    AgentConfig
)
from services.llm_client import GeminiLlmClient, NimLlmClient
from config import settings
from models import (
    ReviewContext,
    Finding,
    ParsedDiff,
    ReviewConfig,
    AnalysisCategory
)

logger = logging.getLogger(__name__)


class SimpleOrchestrator:
    """Simple orchestrator that runs agents directly without LangGraph"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Optional[ReviewConfig] = None
    ):
        """
        Initialize Simple Orchestrator
        
        Args:
            llm: Language model for analyzer agents
            config: Review configuration
        """
        self.llm = llm
        self.config = config or ReviewConfig()
        self.agent_config = AgentConfig()
        
        # Initialize analyzer agents
        self.agents = self._initialize_agents()
        
        logger.info(f"SimpleOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> dict:
        """Initialize all analyzer agents with agent-specific LLM clients"""
        agents = {}
        
        # Check which categories are enabled
        enabled_categories = [
            category.value if hasattr(category, "value") else category
            for category in self.config.enabled_categories
        ]
        
        # Create agent-specific LLM clients if using NVIDIA NIM
        if isinstance(self.llm, GeminiLlmClient):
            if AnalysisCategory.LOGIC.value in enabled_categories:
                agents["logic_analyzer"] = LogicAnalyzerAgent(
                    self.llm.for_agent("logic_analyzer"),
                    self.agent_config
                )
                logger.debug("Logic Analyzer initialized with Gemini")
            
            if AnalysisCategory.READABILITY.value in enabled_categories:
                agents["readability_analyzer"] = ReadabilityAnalyzerAgent(
                    self.llm.for_agent("readability_analyzer"),
                    self.agent_config
                )
                logger.debug("Readability Analyzer initialized with Gemini")
            
            if AnalysisCategory.PERFORMANCE.value in enabled_categories:
                agents["performance_analyzer"] = PerformanceAnalyzerAgent(
                    self.llm.for_agent("performance_analyzer"),
                    self.agent_config
                )
                logger.debug("Performance Analyzer initialized with Gemini")
            
            if AnalysisCategory.SECURITY.value in enabled_categories:
                agents["security_analyzer"] = SecurityAnalyzerAgent(
                    self.llm.for_agent("security_analyzer"),
                    self.agent_config
                )
                logger.debug("Security Analyzer initialized with Gemini")
        elif isinstance(self.llm, NimLlmClient):
            # Each agent gets its own LLM client with the appropriate model
            if AnalysisCategory.LOGIC.value in enabled_categories:
                logic_llm = NimLlmClient(
                    api_key=settings.nvidia_api_key,
                    agent_name="logic_analyzer"
                )
                agents["logic_analyzer"] = LogicAnalyzerAgent(logic_llm, self.agent_config)
                logger.debug(f"Logic Analyzer initialized with model: {logic_llm.model}")
            
            if AnalysisCategory.READABILITY.value in enabled_categories:
                readability_llm = NimLlmClient(
                    api_key=settings.nvidia_api_key,
                    agent_name="readability_analyzer"
                )
                agents["readability_analyzer"] = ReadabilityAnalyzerAgent(readability_llm, self.agent_config)
                logger.debug(f"Readability Analyzer initialized with model: {readability_llm.model}")
            
            if AnalysisCategory.PERFORMANCE.value in enabled_categories:
                performance_llm = NimLlmClient(
                    api_key=settings.nvidia_api_key,
                    agent_name="performance_analyzer"
                )
                agents["performance_analyzer"] = PerformanceAnalyzerAgent(performance_llm, self.agent_config)
                logger.debug(f"Performance Analyzer initialized with model: {performance_llm.model}")
            
            if AnalysisCategory.SECURITY.value in enabled_categories:
                security_llm = NimLlmClient(
                    api_key=settings.nvidia_api_key,
                    agent_name="security_analyzer"
                )
                agents["security_analyzer"] = SecurityAnalyzerAgent(security_llm, self.agent_config)
                logger.debug(f"Security Analyzer initialized with model: {security_llm.model}")
        else:
            # Fallback to using the same LLM for all agents (backward compatibility)
            if AnalysisCategory.LOGIC.value in enabled_categories:
                agents["logic_analyzer"] = LogicAnalyzerAgent(self.llm, self.agent_config)
                logger.debug("Logic Analyzer initialized")
            
            if AnalysisCategory.READABILITY.value in enabled_categories:
                agents["readability_analyzer"] = ReadabilityAnalyzerAgent(self.llm, self.agent_config)
                logger.debug("Readability Analyzer initialized")
            
            if AnalysisCategory.PERFORMANCE.value in enabled_categories:
                agents["performance_analyzer"] = PerformanceAnalyzerAgent(self.llm, self.agent_config)
                logger.debug("Performance Analyzer initialized")
            
            if AnalysisCategory.SECURITY.value in enabled_categories:
                agents["security_analyzer"] = SecurityAnalyzerAgent(self.llm, self.agent_config)
                logger.debug("Security Analyzer initialized")
        
        return agents
    
    async def orchestrate_review(
        self,
        parsed_diff: ParsedDiff,
        pr_metadata: Optional[Any] = None
    ) -> List[Finding]:
        """
        Orchestrate the review process across all agents
        
        Args:
            parsed_diff: Parsed diff with file changes
            pr_metadata: Optional PR metadata
            
        Returns:
            List of findings from all agents
        """
        try:
            # Create review context
            context = ReviewContext(
                file_changes=parsed_diff.files,
                config=self.config,
                pr_metadata=pr_metadata
            )
            
            logger.info(f"Starting review with {len(self.agents)} agents")
            
            # Run all agents concurrently
            tasks = []
            for agent_name, agent in self.agents.items():
                task = self._run_agent(agent_name, agent, context)
                tasks.append(task)
            
            # Wait for all agents to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate findings
            all_findings = []
            for i, result in enumerate(results):
                agent_name = list(self.agents.keys())[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_name} failed: {result}")
                    continue
                
                if result:
                    all_findings.extend(result)
                    logger.info(f"Agent {agent_name} found {len(result)} issues")
            
            # Apply severity filtering
            filtered_findings = self._apply_severity_filter(all_findings)
            
            logger.info(f"Review complete: {len(filtered_findings)} total findings")
            
            return filtered_findings
            
        except Exception as e:
            logger.error(f"Review orchestration failed: {e}", exc_info=True)
            return []
    
    async def _run_agent(
        self,
        agent_name: str,
        agent: Any,
        context: ReviewContext
    ) -> List[Finding]:
        """Run a single agent with timeout"""
        try:
            logger.info(f"Starting {agent_name} analysis")
            start_time = datetime.now()
            
            # Execute with timeout
            findings = await asyncio.wait_for(
                agent.analyze(context),
                timeout=self.agent_config.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{agent_name} completed in {execution_time:.2f}s with {len(findings)} findings")
            
            return findings
            
        except asyncio.TimeoutError:
            logger.error(f"{agent_name} timed out after {self.agent_config.timeout}s")
            return []
        except Exception as e:
            logger.error(f"{agent_name} failed: {e}", exc_info=True)
            return []
    
    def _apply_severity_filter(self, findings: List[Finding]) -> List[Finding]:
        """Apply severity threshold filtering"""
        if not self.config.severity_threshold:
            return findings
        
        # Define severity order
        severity_order = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }
        
        threshold = (
            self.config.severity_threshold.value
            if hasattr(self.config.severity_threshold, "value")
            else self.config.severity_threshold
        )
        threshold_value = severity_order.get(str(threshold).lower(), 1)
        
        # Filter findings
        filtered = [
            f for f in findings
            if severity_order.get(f.severity.value.lower(), 0) >= threshold_value
        ]
        
        if len(filtered) < len(findings):
            logger.info(
                f"Filtered {len(findings) - len(filtered)} findings "
                f"below threshold '{self.config.severity_threshold}'"
            )
        
        return filtered
