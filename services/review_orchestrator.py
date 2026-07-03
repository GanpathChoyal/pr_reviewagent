"""
Review Orchestrator for coordinating multi-agent code review workflow
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseLanguageModel

from agents import (
    LogicAnalyzerAgent,
    ReadabilityAnalyzerAgent,
    PerformanceAnalyzerAgent,
    SecurityAnalyzerAgent,
    AgentConfig
)
from models import (
    ReviewContext,
    Finding,
    ParsedDiff,
    ReviewConfig,
    AnalysisCategory
)

logger = logging.getLogger(__name__)


class ReviewState(Dict[str, Any]):
    """State for the review workflow"""
    pass


class AgentResult:
    """Result from an individual analyzer agent"""
    
    def __init__(
        self,
        agent_name: str,
        findings: List[Finding],
        success: bool = True,
        error: Optional[str] = None,
        execution_time: float = 0.0
    ):
        self.agent_name = agent_name
        self.findings = findings
        self.success = success
        self.error = error
        self.execution_time = execution_time


class ReviewOrchestrator:
    """Orchestrates multi-agent code review workflow using LangGraph"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        config: Optional[ReviewConfig] = None
    ):
        """
        Initialize Review Orchestrator
        
        Args:
            llm: Language model for analyzer agents
            config: Review configuration
        """
        self.llm = llm
        self.config = config or ReviewConfig()
        self.agent_config = AgentConfig()
        
        # Initialize analyzer agents
        self.agents = self._initialize_agents()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info(f"ReviewOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all analyzer agents"""
        agents = {}
        
        # Check which categories are enabled
        enabled_categories = [
            category.value if hasattr(category, "value") else category
            for category in self.config.enabled_categories
        ]
        
        if AnalysisCategory.LOGIC.value in enabled_categories:
            agents["logic_analyzer"] = LogicAnalyzerAgent(
                self._llm_for_agent("logic_analyzer"),
                self.agent_config
            )
            logger.debug("Logic Analyzer initialized")
        
        if AnalysisCategory.READABILITY.value in enabled_categories:
            agents["readability_analyzer"] = ReadabilityAnalyzerAgent(
                self._llm_for_agent("readability_analyzer"),
                self.agent_config
            )
            logger.debug("Readability Analyzer initialized")
        
        if AnalysisCategory.PERFORMANCE.value in enabled_categories:
            agents["performance_analyzer"] = PerformanceAnalyzerAgent(
                self._llm_for_agent("performance_analyzer"),
                self.agent_config
            )
            logger.debug("Performance Analyzer initialized")
        
        if AnalysisCategory.SECURITY.value in enabled_categories:
            agents["security_analyzer"] = SecurityAnalyzerAgent(
                self._llm_for_agent("security_analyzer"),
                self.agent_config
            )
            logger.debug("Security Analyzer initialized")
        
        return agents

    def _llm_for_agent(self, agent_name: str) -> Any:
        """Return an agent-specific LLM client when the provider supports it."""
        if hasattr(self.llm, "for_agent"):
            return self.llm.for_agent(agent_name)
        return self.llm
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for multi-agent orchestration"""
        workflow = StateGraph(ReviewState)
        
        # Add nodes for each enabled agent
        for agent_name in self.agents.keys():
            workflow.add_node(agent_name, self._create_agent_node(agent_name))
        
        # Add aggregator node
        workflow.add_node("aggregator", self._aggregator_node)
        
        # Define parallel execution - all agents start from START
        for agent_name in self.agents.keys():
            workflow.add_edge(START, agent_name)
        
        # All agents feed into aggregator
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "aggregator")
        
        # Aggregator leads to END
        workflow.add_edge("aggregator", END)
        
        return workflow.compile()
    
    def _create_agent_node(self, agent_name: str):
        """Create a node function for an agent"""
        async def agent_node(state: ReviewState) -> ReviewState:
            """Execute agent analysis"""
            try:
                agent = self.agents[agent_name]
                context = state["context"]
                
                logger.info(f"Starting {agent_name} analysis")
                start_time = datetime.now()
                
                # Execute agent analysis with timeout
                findings = await asyncio.wait_for(
                    agent.analyze(context),
                    timeout=self.agent_config.timeout
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Store result in state
                if "agent_results" not in state:
                    state["agent_results"] = []
                
                result = AgentResult(
                    agent_name=agent_name,
                    findings=findings,
                    success=True,
                    execution_time=execution_time
                )
                state["agent_results"].append(result)
                
                logger.info(
                    f"{agent_name} completed: {len(findings)} findings in {execution_time:.2f}s"
                )
                
            except asyncio.TimeoutError:
                logger.error(f"{agent_name} timed out after {self.agent_config.timeout}s")
                if "agent_results" not in state:
                    state["agent_results"] = []
                
                result = AgentResult(
                    agent_name=agent_name,
                    findings=[],
                    success=False,
                    error=f"Timeout after {self.agent_config.timeout}s"
                )
                state["agent_results"].append(result)
                
            except Exception as e:
                logger.error(f"{agent_name} failed: {e}", exc_info=True)
                if "agent_results" not in state:
                    state["agent_results"] = []
                
                result = AgentResult(
                    agent_name=agent_name,
                    findings=[],
                    success=False,
                    error=str(e)
                )
                state["agent_results"].append(result)
            
            return state
        
        return agent_node
    
    def _aggregator_node(self, state: ReviewState) -> ReviewState:
        """Aggregate findings from all agents"""
        try:
            agent_results = state.get("agent_results", [])
            
            # Aggregate findings
            all_findings = self.aggregate_findings(agent_results)
            
            # Apply severity threshold filtering
            filtered_findings = self._apply_severity_filter(all_findings)
            
            state["findings"] = filtered_findings
            state["aggregation_complete"] = True
            
            # Log summary
            successful_agents = sum(1 for r in agent_results if r.success)
            failed_agents = sum(1 for r in agent_results if not r.success)
            
            logger.info(
                f"Aggregation complete: {len(filtered_findings)} findings "
                f"({successful_agents} agents succeeded, {failed_agents} failed)"
            )
            
            # Log failed agents
            for result in agent_results:
                if not result.success:
                    logger.warning(f"Agent {result.agent_name} failed: {result.error}")
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}", exc_info=True)
            state["findings"] = []
            state["aggregation_complete"] = False
            state["aggregation_error"] = str(e)
        
        return state
    
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
            
            logger.info(f"Starting review orchestration with {len(self.agents)} agents")

            agent_results = await self.execute_agents_concurrently(
                list(self.agents.values()),
                context
            )
            findings = self.aggregate_findings(agent_results)
            findings = self._apply_severity_filter(findings)

            successful_agents = sum(1 for result in agent_results if result.success)
            failed_agents = len(agent_results) - successful_agents
            logger.info(
                "Review agents complete: %s succeeded, %s failed",
                successful_agents,
                failed_agents,
            )

            for result in agent_results:
                if not result.success:
                    logger.warning("Agent %s failed: %s", result.agent_name, result.error)
            
            logger.info(f"Review orchestration complete: {len(findings)} total findings")
            
            return findings
            
        except Exception as e:
            logger.error(f"Review orchestration failed: {e}", exc_info=True)
            return []
    
    async def execute_agents_concurrently(
        self,
        agents: List[Any],
        context: ReviewContext
    ) -> List[AgentResult]:
        """
        Execute multiple agents concurrently
        
        Args:
            agents: List of analyzer agents
            context: Review context
            
        Returns:
            List of agent results
        """
        tasks = []
        
        for agent in agents:
            task = self._execute_agent_with_timeout(agent, context)
            tasks.append(task)
        
        # Execute all agents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_name = agents[i].agent_name if hasattr(agents[i], 'agent_name') else f"agent_{i}"
                agent_results.append(AgentResult(
                    agent_name=agent_name,
                    findings=[],
                    success=False,
                    error=str(result)
                ))
            else:
                agent_results.append(result)
        
        return agent_results
    
    async def _execute_agent_with_timeout(
        self,
        agent: Any,
        context: ReviewContext
    ) -> AgentResult:
        """Execute a single agent with timeout"""
        agent_name = agent.agent_name if hasattr(agent, 'agent_name') else "unknown"
        
        try:
            start_time = datetime.now()
            
            # Execute with timeout
            findings = await asyncio.wait_for(
                agent.analyze(context),
                timeout=self.agent_config.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=agent_name,
                findings=findings,
                success=True,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            logger.error(f"{agent_name} timed out")
            return AgentResult(
                agent_name=agent_name,
                findings=[],
                success=False,
                error=f"Timeout after {self.agent_config.timeout}s"
            )
            
        except Exception as e:
            logger.error(f"{agent_name} failed: {e}")
            return AgentResult(
                agent_name=agent_name,
                findings=[],
                success=False,
                error=str(e)
            )
    
    def aggregate_findings(self, agent_results: List[AgentResult]) -> List[Finding]:
        """
        Aggregate findings from all agent results
        
        Args:
            agent_results: List of results from agents
            
        Returns:
            Combined list of findings with agent_source preserved
        """
        all_findings = []
        
        for result in agent_results:
            if result.success and result.findings:
                # Ensure agent_source is set
                for finding in result.findings:
                    if not finding.agent_source:
                        finding.agent_source = result.agent_name
                
                all_findings.extend(result.findings)
                logger.debug(
                    f"Added {len(result.findings)} findings from {result.agent_name}"
                )
        
        logger.info(f"Aggregated {len(all_findings)} total findings")
        
        return all_findings
    
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
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator"""
        return {
            "enabled_agents": list(self.agents.keys()),
            "agent_count": len(self.agents),
            "severity_threshold": self.config.severity_threshold,
            "enabled_categories": self.config.enabled_categories,
            "agent_timeout": self.agent_config.timeout
        }
