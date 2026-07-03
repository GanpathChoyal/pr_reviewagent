"""
Readability Analyzer Agent for evaluating code clarity and maintainability
"""
import logging
import re
from typing import List, Dict, Set

from langchain_core.language_models import BaseLanguageModel

from agents.base_agent import AnalyzerAgent, AgentConfig
from models import Finding, ReviewContext, AnalysisCategory

logger = logging.getLogger(__name__)


class ReadabilityAnalyzerAgent(AnalyzerAgent):
    """Analyzer agent specialized in evaluating code readability and maintainability"""
    
    def __init__(self, llm: BaseLanguageModel, config: AgentConfig = None):
        """
        Initialize Readability Analyzer Agent
        
        Args:
            llm: Language model for analysis
            config: Agent configuration
        """
        super().__init__(llm, config)
        self.agent_name = "readability_analyzer"
    
    def get_analysis_category(self) -> AnalysisCategory:
        """Get the analysis category for this agent"""
        return AnalysisCategory.READABILITY
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for readability analysis"""
        return """You are an expert code reviewer specializing in code readability and maintainability. Your task is to analyze code changes and identify issues that make code harder to read, understand, or maintain.

Focus on detecting these types of readability issues:

1. **Overly Complex Functions**:
   - Functions with high cyclomatic complexity (too many branches/loops)
   - Functions that are too long (>50-100 lines depending on language)
   - Functions doing too many things (violating single responsibility)
   - Deeply nested control structures (>3-4 levels)

2. **Poor Naming Conventions**:
   - Unclear or misleading variable names (x, data, temp, etc.)
   - Inconsistent naming patterns within the same codebase
   - Non-descriptive function names that don't explain purpose
   - Magic numbers without named constants
   - Abbreviations that are hard to understand

3. **Code Structure Issues**:
   - Deeply nested blocks that reduce readability (>3-4 levels)
   - Long parameter lists (>5-7 parameters)
   - Duplicate code that should be extracted
   - Mixed abstraction levels in the same function
   - Poor separation of concerns

4. **Documentation and Comments**:
   - Missing comments for complex logic or algorithms
   - Outdated comments that don't match the code
   - Over-commenting obvious code
   - Missing docstrings for public functions/classes
   - Unclear or misleading comments

5. **Code Organization**:
   - Poor file/module organization
   - Mixing different concerns in the same file
   - Inconsistent code formatting (if not auto-formatted)
   - Unclear import organization
   - Missing type hints (for languages that support them)

6. **Language-Specific Readability**:
   - Not following language idioms and conventions
   - Using outdated patterns when better alternatives exist
   - Inconsistent error handling patterns
   - Not leveraging language features for clarity

**Analysis Guidelines**:
- Focus on maintainability and team collaboration
- Consider the target audience (junior vs senior developers)
- Suggest specific improvements, not just problems
- Be constructive - explain WHY something hurts readability
- Consider the broader codebase context when possible
- Prioritize changes that have the biggest impact on clarity

**Severity Levels**:
- **High**: Significantly impacts code maintainability and team productivity
- **Medium**: Moderately affects readability and could cause confusion
- **Low**: Minor improvements that would enhance clarity

**Output Format**:
Return findings as a JSON array. Each finding MUST include a specific suggestion for improvement:

```json
[
  {
    "file_path": "src/user_service.py",
    "line_number": 25,
    "severity": "medium",
    "description": "Function name 'process_data' is too generic and doesn't describe what kind of processing is done",
    "suggestion": "Rename to something more specific like 'validate_user_input' or 'transform_user_data' based on the actual functionality"
  }
]
```

**Important**: Every finding MUST include a concrete, actionable suggestion. Don't just identify problems - provide solutions.

Return an empty array [] if no readability issues are found."""

    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """
        Analyze code changes for readability issues
        
        Args:
            context: Review context with file changes and configuration
            
        Returns:
            List of findings related to readability issues
        """
        # Validate context
        if not self.validate_context(context):
            logger.debug("Context validation failed for readability analysis")
            return []
        
        try:
            # Pre-analyze code for complexity metrics
            complexity_insights = self._analyze_complexity(context)
            
            # Create prompt messages
            messages = self.create_prompt(context)
            
            # Add readability-specific context
            enhanced_messages = self._enhance_prompt_with_readability_context(
                messages, context, complexity_insights
            )
            
            # Invoke LLM with retry
            response = await self._invoke_llm_with_retry(enhanced_messages)
            
            # Parse response into findings
            findings = await self.parse_llm_response(response)
            
            # Filter and validate findings
            validated_findings = self._validate_readability_findings(findings, context)
            
            logger.info(f"Readability analyzer found {len(validated_findings)} issues")
            return validated_findings
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return []
    
    def _analyze_complexity(self, context: ReviewContext) -> Dict[str, any]:
        """Pre-analyze code for complexity metrics"""
        insights = {
            "long_functions": [],
            "deep_nesting": [],
            "long_parameter_lists": [],
            "magic_numbers": [],
            "poor_names": []
        }
        
        for file_change in context.file_changes:
            if file_change.is_binary:
                continue
            
            # Analyze additions for complexity
            for line_change in file_change.additions:
                line_content = line_change.content.strip()
                
                # Check for deep nesting (count leading whitespace)
                if line_content:
                    indent_level = self._count_indent_level(line_change.content, file_change.language)
                    if indent_level > 4:
                        insights["deep_nesting"].append({
                            "file": file_change.file_path,
                            "line": line_change.line_number,
                            "level": indent_level
                        })
                
                # Check for magic numbers
                magic_numbers = self._find_magic_numbers(line_content, file_change.language)
                for number in magic_numbers:
                    insights["magic_numbers"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "number": number
                    })
                
                # Check for poor variable names
                poor_names = self._find_poor_names(line_content, file_change.language)
                for name in poor_names:
                    insights["poor_names"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "name": name
                    })
                
                # Check for long parameter lists
                if self._is_function_definition(line_content, file_change.language):
                    param_count = self._count_parameters(line_content, file_change.language)
                    if param_count > 5:
                        insights["long_parameter_lists"].append({
                            "file": file_change.file_path,
                            "line": line_change.line_number,
                            "count": param_count
                        })
        
        return insights
    
    def _count_indent_level(self, line: str, language: str) -> int:
        """Count indentation level of a line"""
        if language == "python":
            # Python uses spaces for indentation
            return (len(line) - len(line.lstrip())) // 4
        else:
            # Most other languages use braces, count leading whitespace
            return len(line) - len(line.lstrip())
    
    def _find_magic_numbers(self, line: str, language: str) -> List[str]:
        """Find magic numbers in a line of code"""
        magic_numbers = []
        
        # Common patterns for numbers that might be magic
        number_patterns = [
            r'\b(\d{2,})\b',  # Numbers with 2+ digits
            r'\b(0x[0-9a-fA-F]+)\b',  # Hex numbers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                # Skip common non-magic numbers
                if match not in ['0', '1', '2', '10', '100', '1000']:
                    # Skip if it looks like it's in a comment
                    if '//' not in line[:line.find(match)] and '#' not in line[:line.find(match)]:
                        magic_numbers.append(match)
        
        return magic_numbers
    
    def _find_poor_names(self, line: str, language: str) -> List[str]:
        """Find poorly named variables in a line of code"""
        poor_names = []
        
        # Common poor variable names
        bad_names = {
            'x', 'y', 'z', 'i', 'j', 'k', 'n', 'm',  # Single letters (except in loops)
            'data', 'info', 'item', 'obj', 'thing',   # Too generic
            'temp', 'tmp', 'val', 'var', 'foo', 'bar' # Temporary/placeholder names
        }
        
        # Simple pattern to find variable assignments
        if language == "python":
            # Python variable assignment
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        elif language in ["javascript", "typescript"]:
            # JavaScript variable declaration
            var_pattern = r'(?:let|const|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        elif language == "java":
            # Java variable declaration (simplified)
            var_pattern = r'\b[A-Z][a-zA-Z]*\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        else:
            # Generic pattern
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        
        matches = re.findall(var_pattern, line)
        for match in matches:
            if match.lower() in bad_names:
                poor_names.append(match)
        
        return poor_names
    
    def _is_function_definition(self, line: str, language: str) -> bool:
        """Check if line contains a function definition"""
        if language == "python":
            return line.strip().startswith('def ')
        elif language in ["javascript", "typescript"]:
            return 'function' in line or '=>' in line
        elif language == "java":
            return ('public' in line or 'private' in line or 'protected' in line) and '(' in line
        elif language in ["cpp", "c"]:
            return '(' in line and ')' in line and '{' not in line.split('(')[0]
        else:
            return '(' in line and ')' in line
    
    def _count_parameters(self, line: str, language: str) -> int:
        """Count parameters in a function definition"""
        try:
            # Find content between parentheses
            start = line.find('(')
            end = line.rfind(')')
            if start == -1 or end == -1 or start >= end:
                return 0
            
            params_str = line[start+1:end].strip()
            if not params_str:
                return 0
            
            # Simple comma counting (not perfect but good enough)
            return len([p for p in params_str.split(',') if p.strip()])
        except:
            return 0
    
    def _enhance_prompt_with_readability_context(self, messages, context: ReviewContext, complexity_insights: Dict):
        """Enhance prompt with readability-specific context"""
        # Get the human message (last message)
        human_message = messages[-1]
        
        # Add readability-specific instructions
        readability_context = self._build_readability_context(context, complexity_insights)
        
        enhanced_content = f"{human_message.content}\n\n{readability_context}"
        
        # Replace the human message with enhanced version
        from langchain_core.messages import HumanMessage
        messages[-1] = HumanMessage(content=enhanced_content)
        
        return messages
    
    def _build_readability_context(self, context: ReviewContext, complexity_insights: Dict) -> str:
        """Build readability-specific analysis context"""
        context_parts = []
        
        # Language-specific readability guidelines
        languages = set(f.language for f in context.file_changes if not f.is_binary)
        if languages:
            context_parts.append("**Language-Specific Readability Guidelines:**")
            
            for lang in languages:
                if lang == "python":
                    context_parts.append("- Python: Follow PEP 8, use descriptive names, prefer list comprehensions, add type hints")
                elif lang == "javascript":
                    context_parts.append("- JavaScript: Use const/let, descriptive function names, avoid callback hell, use modern ES6+ features")
                elif lang == "java":
                    context_parts.append("- Java: Follow camelCase, use meaningful class/method names, avoid deep inheritance")
                elif lang == "cpp" or lang == "c":
                    context_parts.append("- C/C++: Use clear variable names, avoid macros when possible, consistent formatting")
                elif lang == "go":
                    context_parts.append("- Go: Follow Go conventions, use short but clear names, prefer composition over inheritance")
        
        # Complexity insights
        if any(complexity_insights.values()):
            context_parts.append("\n**Pre-Analysis Findings:**")
            
            if complexity_insights["deep_nesting"]:
                count = len(complexity_insights["deep_nesting"])
                context_parts.append(f"- Found {count} instances of deep nesting (>4 levels)")
            
            if complexity_insights["magic_numbers"]:
                count = len(complexity_insights["magic_numbers"])
                context_parts.append(f"- Found {count} potential magic numbers")
            
            if complexity_insights["poor_names"]:
                count = len(complexity_insights["poor_names"])
                context_parts.append(f"- Found {count} potentially poor variable names")
            
            if complexity_insights["long_parameter_lists"]:
                count = len(complexity_insights["long_parameter_lists"])
                context_parts.append(f"- Found {count} functions with >5 parameters")
        
        # File context
        file_types = self._categorize_files(context.file_changes)
        if file_types:
            context_parts.append(f"\n**File Types**: {', '.join(file_types)} - adjust readability standards accordingly")
        
        # Change context
        total_additions = sum(len(f.additions) for f in context.file_changes)
        total_deletions = sum(len(f.deletions) for f in context.file_changes)
        
        if total_additions > total_deletions * 2:
            context_parts.append("\n**Focus**: Primarily new code - emphasize establishing good patterns")
        elif total_deletions > total_additions:
            context_parts.append("\n**Focus**: Code removal - check if remaining code is still clear")
        else:
            context_parts.append("\n**Focus**: Mixed changes - ensure modifications maintain readability")
        
        return "\n".join(context_parts)
    
    def _categorize_files(self, file_changes) -> Set[str]:
        """Categorize files by type for context"""
        categories = set()
        
        for file_change in file_changes:
            path = file_change.file_path.lower()
            
            if any(keyword in path for keyword in ['test', 'spec']):
                categories.add("test")
            elif any(keyword in path for keyword in ['config', 'setting']):
                categories.add("configuration")
            elif any(keyword in path for keyword in ['util', 'helper', 'common']):
                categories.add("utility")
            elif any(keyword in path for keyword in ['model', 'entity', 'dto']):
                categories.add("data model")
            elif any(keyword in path for keyword in ['service', 'controller', 'handler']):
                categories.add("business logic")
            elif any(keyword in path for keyword in ['ui', 'view', 'component']):
                categories.add("user interface")
            else:
                categories.add("application")
        
        return categories
    
    def _validate_readability_findings(self, findings: List[Finding], context: ReviewContext) -> List[Finding]:
        """Validate and filter readability-specific findings"""
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
            
            # Validate description is meaningful
            if len(finding.description.strip()) < 15:
                logger.warning(f"Finding description too short: {finding.description}")
                continue
            
            # Ensure suggestion is provided (required for readability findings)
            if not finding.suggestion or len(finding.suggestion.strip()) < 10:
                logger.warning(f"Missing or insufficient suggestion for readability finding: {finding.suggestion}")
                continue
            
            # Ensure category is correct
            finding.category = AnalysisCategory.READABILITY
            finding.agent_source = self.agent_name
            
            validated.append(finding)
        
        return validated
    
    def get_readability_patterns(self) -> Dict[str, List[str]]:
        """Get readability patterns by language"""
        return {
            "python": [
                "PEP 8 compliance",
                "Descriptive variable names",
                "Function length (<50 lines)",
                "Type hints usage",
                "List comprehensions over loops",
                "Docstring presence"
            ],
            "javascript": [
                "Consistent naming (camelCase)",
                "Modern ES6+ features",
                "Avoiding callback hell",
                "Clear function purposes",
                "Proper error handling",
                "JSDoc comments"
            ],
            "java": [
                "CamelCase conventions",
                "Single responsibility principle",
                "Meaningful class/method names",
                "Proper encapsulation",
                "Javadoc documentation",
                "Avoiding deep inheritance"
            ],
            "cpp": [
                "Consistent naming conventions",
                "Clear variable names",
                "Avoiding complex macros",
                "RAII principles",
                "Header documentation",
                "Const correctness"
            ],
            "go": [
                "Go naming conventions",
                "Short but clear names",
                "Package organization",
                "Error handling patterns",
                "Interface usage",
                "Go doc comments"
            ]
        }
    
    def get_agent_info(self) -> Dict[str, any]:
        """Get information about this agent"""
        info = super().get_agent_info()
        info.update({
            "specialization": "Code readability and maintainability",
            "evaluates": [
                "Function complexity",
                "Naming conventions",
                "Code structure",
                "Documentation quality",
                "Code organization",
                "Language idioms"
            ],
            "supported_patterns": list(self.get_readability_patterns().keys()),
            "requires_suggestions": True
        })
        return info