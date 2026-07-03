"""
Base classes and shared helpers for analyzer agents.
"""
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

from models import AnalysisCategory, Finding, ReviewContext, SeverityLevel

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Runtime settings shared by analyzer agents."""

    timeout: int = 120
    max_retries: int = 2
    retry_delay: float = 1.0
    max_prompt_chars: int = 24000


class AnalyzerAgent(ABC):
    """Base class for specialized code-review analyzer agents."""

    def __init__(self, llm: BaseLanguageModel, config: AgentConfig = None):
        self.llm = llm
        self.config = config or AgentConfig()
        self.agent_name = self.__class__.__name__

    @abstractmethod
    def get_analysis_category(self) -> AnalysisCategory:
        """Return the category this analyzer is responsible for."""

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt used for this analyzer."""

    @abstractmethod
    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """Analyze the supplied review context."""

    def validate_context(self, context: ReviewContext) -> bool:
        """Check that the analyzer received useful, non-binary changes."""
        if not context or not context.file_changes:
            return False

        return any(not file_change.is_binary for file_change in context.file_changes)

    def create_prompt(self, context: ReviewContext) -> List[Any]:
        """Build LangChain messages from the review context."""
        diff_text = self._format_context(context)
        if len(diff_text) > self.config.max_prompt_chars:
            diff_text = diff_text[: self.config.max_prompt_chars]
            diff_text += "\n\n[Diff truncated because it exceeded prompt size limits.]"

        human_prompt = (
            "Review the following pull request changes. "
            "Return only a JSON array of findings in the requested schema.\n\n"
            f"{diff_text}"
        )

        return [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=human_prompt),
        ]

    async def _invoke_llm_with_retry(self, messages: Sequence[Any]) -> Any:
        """Invoke the configured LLM with simple retry handling."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.llm.ainvoke(messages)
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break

                delay = self.config.retry_delay * (attempt + 1)
                logger.warning(
                    "%s LLM call failed, retrying in %.1fs: %s",
                    self.agent_name,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        raise last_error

    async def parse_llm_response(self, response: Any) -> List[Finding]:
        """Parse an LLM JSON response into Finding objects."""
        content = self._response_to_text(response)
        data = self._extract_json_array(content)
        findings = []

        for item in data:
            if not isinstance(item, dict):
                logger.warning("Skipping non-object finding from %s: %r", self.agent_name, item)
                continue

            try:
                item.setdefault("category", self.get_analysis_category())
                item.setdefault("agent_source", self.agent_name)
                item["severity"] = self._normalize_severity(item.get("severity"))
                findings.append(Finding(**item))
            except Exception as exc:
                logger.warning("Skipping invalid finding from %s: %s", self.agent_name, exc)

        return findings

    def get_agent_info(self) -> dict:
        """Return basic metadata about this analyzer."""
        return {
            "agent_name": self.agent_name,
            "category": self.get_analysis_category().value,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

    def _format_context(self, context: ReviewContext) -> str:
        """Format changed files and lines into a compact prompt section."""
        sections = []

        if context.pr_metadata:
            sections.append(
                "Pull Request:\n"
                f"- repository: {context.pr_metadata.repository}\n"
                f"- number: {context.pr_metadata.pr_number}\n"
                f"- title: {context.pr_metadata.title}"
            )

        for file_change in context.file_changes:
            if file_change.is_binary:
                continue

            lines = [
                f"File: {file_change.file_path}",
                f"Language: {file_change.language}",
            ]

            for change in file_change.additions:
                lines.append(f"+{change.line_number}: {change.content}")
            for change in file_change.deletions:
                lines.append(f"-{change.line_number}: {change.content}")
            for change in file_change.modifications:
                lines.append(f"~{change.line_number}: {change.content}")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _response_to_text(self, response: Any) -> str:
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _extract_json_array(self, content: str) -> List[Any]:
        content = content.strip()
        if not content:
            return []

        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])

        return []

    def _normalize_severity(self, severity: Any) -> str:
        value = str(severity or SeverityLevel.LOW.value).lower()
        if value not in {level.value for level in SeverityLevel}:
            return SeverityLevel.LOW.value
        return value
