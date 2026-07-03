"""
Logic Analyzer Agent for detecting logical errors and bugs
"""
import logging
from typing import List

from langchain_core.language_models import BaseLanguageModel

from agents.base_agent import AnalyzerAgent, AgentConfig
from models import Finding, ReviewContext, AnalysisCategory

logger = logging.getLogger(__name__)


class LogicAnalyzerAgent(AnalyzerAgent):
    """Analyzer agent specialized in detecting logical errors and bugs"""
    
    def __init__(self, llm: BaseLanguageModel, config: AgentConfig = None):
        """
        Initialize Logic Analyzer Agent
        
        Args:
            llm: Language model for analysis
            config: Agent configuration
        """
        super().__init__(llm, config)
        self.agent_name = "logic_analyzer"
    
    def get_analysis_category(self) -> AnalysisCategory:
        """Get the analysis category for this agent"""
        return AnalysisCategory.LOGIC
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for logic analysis"""
        return """You are an expert code reviewer specializing in logical errors and bugs. Your task is to analyze code changes and identify potential logical issues that could cause runtime errors, incorrect behavior, or crashes.

Focus on detecting these types of logical issues:

1. **Null Pointer Dereferences**:
   - Accessing properties/methods on potentially null/undefined variables
   - Missing null checks before object access
   - Dereferencing pointers without validation

2. **Unreachable Code**:
   - Code after return statements
   - Code in impossible conditional branches
   - Dead code that can never be executed

3. **Infinite Loops and Off-by-One Errors**:
   - Loop conditions that never become false
   - Incorrect loop bounds (< vs <=, > vs >=)
   - Missing loop variable updates
   - Array/list index out of bounds

4. **Incorrect Parameter Usage**:
   - Wrong number of arguments in function calls
   - Type mismatches in parameters
   - Missing required parameters
   - Incorrect parameter order

5. **Logic Flow Issues**:
   - Missing return statements in non-void functions
   - Incorrect conditional logic (wrong operators, missing conditions)
   - Race conditions in concurrent code
   - Resource leaks (unclosed files, connections, etc.)

6. **Data Type Issues**:
   - Integer overflow/underflow
   - Division by zero
   - Incorrect type conversions
   - String/array bounds violations

7. **Exception Handling**:
   - Missing error handling for risky operations
   - Catching too broad exceptions
   - Not handling specific error cases

**Analysis Guidelines**:
- Focus ONLY on logical correctness, not style or performance
- Provide specific line numbers where issues occur
- Explain WHY the code is problematic
- Suggest concrete fixes when possible
- Consider the programming language context
- Be conservative - only flag clear logical errors

**Severity Levels**:
- **Critical**: Will definitely cause crashes or data corruption
- **High**: Likely to cause runtime errors or incorrect behavior
- **Medium**: Could cause issues under certain conditions
- **Low**: Minor logical inconsistencies

**Output Format**:
Return findings as a JSON array. Each finding must include:
- file_path: The file containing the issue
- line_number: The specific line with the problem
- severity: One of "critical", "high", "medium", "low"
- description: Clear explanation of the logical issue
- suggestion: Specific recommendation for fixing the issue

Example:
```json
[
  {
    "file_path": "src/calculator.py",
    "line_number": 15,
    "severity": "critical",
    "description": "Division by zero error: variable 'denominator' is not checked for zero before division",
    "suggestion": "Add a check: if denominator == 0: raise ValueError('Division by zero')"
  }
]
```

Return an empty array [] if no logical issues are found."""

    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """
        Analyze code changes for logical errors
        
        Args:
            context: Review context with file changes and configuration
            
        Returns:
            List of findings related to logical issues
        """
        # Validate context
        if not self.validate_context(context):
            logger.debug("Context validation failed for logic analysis")
            return []
        
        try:
            # Create prompt messages
            messages = self.create_prompt(context)
            
            # Add logic-specific context to the human message
            enhanced_messages = self._enhance_prompt_with_logic_context(messages, context)
            
            # Invoke LLM with retry
            response = await self._invoke_llm_with_retry(enhanced_messages)
            
            # Parse response into findings
            findings = await self.parse_llm_response(response)
            
            # Filter and validate findings
            validated_findings = self._validate_logic_findings(findings, context)
            
            logger.info(f"Logic analyzer found {len(validated_findings)} issues")
            return validated_findings
            
        except Exception as e:
            logger.error(f"Logic analysis failed: {e}")
            return []
    
    def _enhance_prompt_with_logic_context(self, messages, context: ReviewContext):
        """Enhance prompt with logic-specific context"""
        # Get the human message (last message)
        human_message = messages[-1]
        
        # Add logic-specific instructions
        logic_context = self._build_logic_context(context)
        
        enhanced_content = f"{human_message.content}\n\n{logic_context}"
        
        # Replace the human message with enhanced version
        from langchain_core.messages import HumanMessage
        messages[-1] = HumanMessage(content=enhanced_content)
        
        return messages
    
    def _build_logic_context(self, context: ReviewContext) -> str:
        """Build logic-specific analysis context"""
        context_parts = []
        
        # Language-specific guidance
        languages = set(f.language for f in context.file_changes if not f.is_binary)
        if languages:
            context_parts.append("**Language-Specific Considerations:**")
            
            for lang in languages:
                if lang == "python":
                    context_parts.append("- Python: Watch for None checks, list/dict key errors, indentation logic")
                elif lang == "javascript":
                    context_parts.append("- JavaScript: Check for undefined/null, type coercion issues, async/await problems")
                elif lang == "java":
                    context_parts.append("- Java: Look for NullPointerException, array bounds, resource management")
                elif lang == "cpp" or lang == "c":
                    context_parts.append("- C/C++: Check pointer dereferencing, memory management, buffer overflows")
                elif lang == "go":
                    context_parts.append("- Go: Watch for nil pointer dereference, goroutine race conditions")
                elif lang == "rust":
                    context_parts.append("- Rust: Focus on Option/Result handling, borrowing issues")
        
        # Analysis focus based on change types
        has_additions = any(f.additions for f in context.file_changes)
        has_deletions = any(f.deletions for f in context.file_changes)
        
        if has_additions and has_deletions:
            context_parts.append("\n**Focus Areas**: Pay special attention to modified logic flows and new code paths")
        elif has_additions:
            context_parts.append("\n**Focus Areas**: Analyze new code for logical correctness and error handling")
        elif has_deletions:
            context_parts.append("\n**Focus Areas**: Check if deletions break existing logic or create unreachable code")
        
        # File-specific context
        file_types = set()
        for file_change in context.file_changes:
            if not file_change.is_binary:
                if "test" in file_change.file_path.lower():
                    file_types.add("test")
                elif any(keyword in file_change.file_path.lower() for keyword in ["config", "setting"]):
                    file_types.add("config")
                elif any(keyword in file_change.file_path.lower() for keyword in ["util", "helper"]):
                    file_types.add("utility")
                else:
                    file_types.add("application")
        
        if file_types:
            context_parts.append(f"\n**File Types**: {', '.join(file_types)} - adjust analysis accordingly")
        
        return "\n".join(context_parts)
    
    def _validate_logic_findings(self, findings: List[Finding], context: ReviewContext) -> List[Finding]:
        """Validate and filter logic-specific findings"""
        validated = []
        
        for finding in findings:
            # Ensure finding is in a file that was actually changed
            file_exists = any(
                f.file_path == finding.file_path 
                for f in context.file_changes
            )
            
            if not file_exists:
                logger.warning(f"Finding references non-existent file: {finding.file_path}")
                continue
            
            # Validate line number is reasonable
            if finding.line_number <= 0:
                logger.warning(f"Invalid line number: {finding.line_number}")
                continue
            
            # Check if line number corresponds to actual changes
            file_change = next(
                (f for f in context.file_changes if f.file_path == finding.file_path),
                None
            )
            
            if file_change and not file_change.is_binary:
                # Check if the line number is in the changed lines
                changed_lines = set()
                
                # Add all addition line numbers
                changed_lines.update(line.line_number for line in file_change.additions)
                
                # Add all modification line numbers
                changed_lines.update(line.line_number for line in file_change.modifications)
                
                # For deletions, we can't validate line numbers as easily
                # but we should be more lenient
                if changed_lines and finding.line_number not in changed_lines:
                    # Allow some tolerance for nearby lines (context lines)
                    nearby_lines = any(
                        abs(finding.line_number - line_num) <= 3 
                        for line_num in changed_lines
                    )
                    
                    if not nearby_lines:
                        logger.debug(f"Finding line {finding.line_number} not in changed lines for {finding.file_path}")
                        # Still include it, but log for debugging
            
            # Validate description is meaningful
            if len(finding.description.strip()) < 10:
                logger.warning(f"Finding description too short: {finding.description}")
                continue
            
            # Ensure category is correct
            finding.category = AnalysisCategory.LOGIC
            finding.agent_source = self.agent_name
            
            validated.append(finding)
        
        return validated
    
    def get_logic_patterns(self) -> dict:
        """Get common logic error patterns by language"""
        return {
            "python": [
                "None checks",
                "List/dict key errors", 
                "Division by zero",
                "Index out of bounds",
                "Type errors"
            ],
            "javascript": [
                "Undefined/null checks",
                "Type coercion issues",
                "Async/await problems",
                "Callback errors",
                "Scope issues"
            ],
            "java": [
                "NullPointerException",
                "Array bounds",
                "Resource leaks",
                "Concurrent access",
                "Type casting"
            ],
            "cpp": [
                "Null pointer dereference",
                "Memory leaks",
                "Buffer overflows",
                "Use after free",
                "Uninitialized variables"
            ],
            "go": [
                "Nil pointer dereference",
                "Race conditions",
                "Channel deadlocks",
                "Error handling",
                "Slice bounds"
            ]
        }
    
    def get_agent_info(self) -> dict:
        """Get information about this agent"""
        info = super().get_agent_info()
        info.update({
            "specialization": "Logical errors and bugs",
            "detects": [
                "Null pointer dereferences",
                "Unreachable code",
                "Infinite loops",
                "Off-by-one errors",
                "Parameter misuse",
                "Logic flow issues",
                "Exception handling gaps"
            ],
            "supported_patterns": list(self.get_logic_patterns().keys())
        })
        return info