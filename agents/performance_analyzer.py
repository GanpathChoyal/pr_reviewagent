"""
Performance Analyzer Agent for identifying performance issues and optimization opportunities
"""
import logging
import re
from typing import List, Dict, Set, Tuple

from langchain_core.language_models import BaseLanguageModel

from agents.base_agent import AnalyzerAgent, AgentConfig
from models import Finding, ReviewContext, AnalysisCategory

logger = logging.getLogger(__name__)


class PerformanceAnalyzerAgent(AnalyzerAgent):
    """Analyzer agent specialized in identifying performance issues and optimization opportunities"""
    
    def __init__(self, llm: BaseLanguageModel, config: AgentConfig = None):
        """
        Initialize Performance Analyzer Agent
        
        Args:
            llm: Language model for analysis
            config: Agent configuration
        """
        super().__init__(llm, config)
        self.agent_name = "performance_analyzer"
    
    def get_analysis_category(self) -> AnalysisCategory:
        """Get the analysis category for this agent"""
        return AnalysisCategory.PERFORMANCE
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for performance analysis"""
        return """You are an expert code reviewer specializing in performance optimization. Your task is to analyze code changes and identify performance issues, inefficiencies, and optimization opportunities that could impact application speed, memory usage, or scalability.

Focus on detecting these types of performance issues:

1. **Inefficient Algorithms and Time Complexity**:
   - O(n²) or worse algorithms where better alternatives exist
   - Unnecessary nested loops that could be optimized
   - Inefficient sorting or searching approaches
   - Recursive algorithms without memoization where appropriate
   - Brute force solutions where optimized algorithms exist

2. **Inappropriate Data Structure Choices**:
   - Using lists where sets/dictionaries would be more efficient for lookups
   - Linear searches in large collections instead of indexed lookups
   - Inefficient data structure operations (e.g., frequent insertions at list beginning)
   - Missing caching or memoization for expensive computations
   - Inappropriate collection types for the use case

3. **Redundant Computations and Caching Issues**:
   - Repeated expensive calculations that could be cached
   - Computing the same values multiple times in loops
   - Missing memoization for recursive functions
   - Inefficient string concatenation in loops
   - Redundant database queries or API calls

4. **Database and I/O Performance Issues**:
   - N+1 query patterns (querying in loops)
   - Missing database indexes for frequent queries
   - Loading unnecessary data (SELECT * instead of specific fields)
   - Synchronous I/O operations that could be asynchronous
   - Inefficient batch operations
   - Missing connection pooling or reuse

5. **Memory and Resource Management**:
   - Memory leaks or excessive memory usage
   - Creating unnecessary objects in loops
   - Not releasing resources properly
   - Inefficient memory allocation patterns
   - Large object creation that could be avoided

6. **Concurrency and Parallelization Issues**:
   - Missing opportunities for parallel processing
   - Inefficient synchronization mechanisms
   - Blocking operations that could be non-blocking
   - Race conditions that hurt performance
   - Suboptimal thread pool usage

7. **Language-Specific Performance Anti-patterns**:
   - Python: Not using list comprehensions, inefficient string operations
   - JavaScript: Blocking the event loop, inefficient DOM operations
   - Java: Autoboxing overhead, inefficient collections usage
   - C++: Unnecessary copying, missing move semantics
   - Go: Inefficient goroutine usage, missing buffered channels

**Analysis Guidelines**:
- Focus on measurable performance impacts, not micro-optimizations
- Consider the scale and context (performance critical vs. one-time operations)
- Explain the performance impact in concrete terms (time/space complexity)
- Suggest specific optimizations with expected improvements
- Consider trade-offs between performance and readability/maintainability
- Prioritize issues that affect user experience or system scalability

**Severity Levels**:
- **High**: Significant performance impact, affects user experience or scalability
- **Medium**: Noticeable performance impact under load or with large datasets
- **Low**: Minor optimization opportunities with measurable but small impact

**Output Format**:
Return findings as a JSON array. Each finding MUST include performance impact explanation and optimization suggestion:

```json
[
  {
    "file_path": "src/data_service.py",
    "line_number": 25,
    "severity": "high",
    "description": "N+1 query pattern: executing database query inside loop will result in O(n) database calls instead of O(1)",
    "suggestion": "Move query outside loop and use batch loading: users = User.objects.filter(id__in=user_ids) then create lookup dictionary"
  }
]
```

**Important**: Every finding MUST explain the performance impact and provide a specific optimization strategy with expected improvement.

Return an empty array [] if no performance issues are found."""

    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """
        Analyze code changes for performance issues
        
        Args:
            context: Review context with file changes and configuration
            
        Returns:
            List of findings related to performance issues
        """
        # Validate context
        if not self.validate_context(context):
            logger.debug("Context validation failed for performance analysis")
            return []
        
        try:
            # Pre-analyze code for performance patterns
            performance_insights = self._analyze_performance_patterns(context)
            
            # Create prompt messages
            messages = self.create_prompt(context)
            
            # Add performance-specific context
            enhanced_messages = self._enhance_prompt_with_performance_context(
                messages, context, performance_insights
            )
            
            # Invoke LLM with retry
            response = await self._invoke_llm_with_retry(enhanced_messages)
            
            # Parse response into findings
            findings = await self.parse_llm_response(response)
            
            # Filter and validate findings
            validated_findings = self._validate_performance_findings(findings, context)
            
            logger.info(f"Performance analyzer found {len(validated_findings)} issues")
            return validated_findings
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return []
    
    def _analyze_performance_patterns(self, context: ReviewContext) -> Dict[str, any]:
        """Pre-analyze code for performance anti-patterns"""
        insights = {
            "nested_loops": [],
            "database_queries_in_loops": [],
            "inefficient_data_structures": [],
            "string_concatenation_in_loops": [],
            "redundant_computations": [],
            "synchronous_io": [],
            "missing_caching": []
        }
        
        for file_change in context.file_changes:
            if file_change.is_binary:
                continue
            
            # Analyze additions for performance patterns
            for line_change in file_change.additions:
                line_content = line_change.content.strip()
                
                if not line_content:
                    continue
                
                # Check for nested loops
                if self._is_loop_start(line_content, file_change.language):
                    nested_level = self._count_loop_nesting(
                        file_change.additions, line_change.line_number, file_change.language
                    )
                    if nested_level > 1:
                        insights["nested_loops"].append({
                            "file": file_change.file_path,
                            "line": line_change.line_number,
                            "nesting_level": nested_level
                        })
                
                # Check for database queries in loops
                if self._contains_database_query(line_content, file_change.language):
                    if self._is_inside_loop(file_change.additions, line_change.line_number, file_change.language):
                        insights["database_queries_in_loops"].append({
                            "file": file_change.file_path,
                            "line": line_change.line_number,
                            "query_type": self._identify_query_type(line_content)
                        })
                
                # Check for inefficient data structure usage
                inefficient_patterns = self._find_inefficient_data_structure_usage(line_content, file_change.language)
                for pattern in inefficient_patterns:
                    insights["inefficient_data_structures"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "pattern": pattern
                    })
                
                # Check for string concatenation in loops
                if self._is_string_concatenation(line_content, file_change.language):
                    if self._is_inside_loop(file_change.additions, line_change.line_number, file_change.language):
                        insights["string_concatenation_in_loops"].append({
                            "file": file_change.file_path,
                            "line": line_change.line_number
                        })
                
                # Check for synchronous I/O operations
                if self._is_synchronous_io(line_content, file_change.language):
                    insights["synchronous_io"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "operation": self._identify_io_operation(line_content)
                    })
        
        return insights
    
    def _is_loop_start(self, line: str, language: str) -> bool:
        """Check if line starts a loop"""
        if language == "python":
            return line.strip().startswith(('for ', 'while '))
        elif language in ["javascript", "typescript"]:
            return any(pattern in line for pattern in ['for (', 'for(', 'while (', 'while(', 'forEach('])
        elif language == "java":
            return any(pattern in line for pattern in ['for (', 'for(', 'while (', 'while('])
        elif language in ["cpp", "c"]:
            return any(pattern in line for pattern in ['for (', 'for(', 'while (', 'while('])
        elif language == "go":
            return line.strip().startswith('for ')
        else:
            return any(keyword in line.lower() for keyword in ['for', 'while'])
    
    def _count_loop_nesting(self, additions: List, current_line: int, language: str) -> int:
        """Count nesting level of loops"""
        nesting = 0
        indent_level = 0
        
        # Look backwards from current line to count nesting
        for line_change in reversed(additions):
            if line_change.line_number >= current_line:
                continue
            
            line_content = line_change.content
            current_indent = len(line_content) - len(line_content.lstrip())
            
            if self._is_loop_start(line_content, language):
                if current_indent < indent_level or indent_level == 0:
                    nesting += 1
                    indent_level = current_indent
        
        return nesting
    
    def _contains_database_query(self, line: str, language: str) -> bool:
        """Check if line contains a database query"""
        db_patterns = [
            # SQL-like patterns
            r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b',
            # ORM patterns
            r'\.(find|findOne|findAll|save|create|update|delete|query|execute)\(',
            r'\.(filter|get|all|first|last)\(',
            # Database connection patterns
            r'\.(cursor|execute|fetchone|fetchall|commit)\(',
            # Framework-specific patterns
            r'(query|exec|prepare)\(',
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in db_patterns)
    
    def _is_inside_loop(self, additions: List, current_line: int, language: str) -> bool:
        """Check if current line is inside a loop"""
        current_indent = None
        
        # Find current line's indentation
        for line_change in additions:
            if line_change.line_number == current_line:
                current_indent = len(line_change.content) - len(line_change.content.lstrip())
                break
        
        if current_indent is None:
            return False
        
        # Look backwards for loop starts with less indentation
        for line_change in reversed(additions):
            if line_change.line_number >= current_line:
                continue
            
            line_content = line_change.content
            line_indent = len(line_content) - len(line_content.lstrip())
            
            if line_indent < current_indent and self._is_loop_start(line_content, language):
                return True
        
        return False
    
    def _identify_query_type(self, line: str) -> str:
        """Identify the type of database query"""
        if re.search(r'\b(SELECT|find|get|filter)\b', line, re.IGNORECASE):
            return "SELECT/READ"
        elif re.search(r'\b(INSERT|create|save)\b', line, re.IGNORECASE):
            return "INSERT/CREATE"
        elif re.search(r'\b(UPDATE|update)\b', line, re.IGNORECASE):
            return "UPDATE"
        elif re.search(r'\b(DELETE|delete|remove)\b', line, re.IGNORECASE):
            return "DELETE"
        else:
            return "UNKNOWN"
    
    def _find_inefficient_data_structure_usage(self, line: str, language: str) -> List[str]:
        """Find inefficient data structure usage patterns"""
        patterns = []
        
        # Linear search in collections
        if re.search(r'\bin\s+\w+\s*\[', line) or ' in ' in line:
            if language == "python" and not re.search(r'\bset\(|\bdict\(', line):
                patterns.append("linear_search_in_list")
        
        # Inefficient list operations
        if language == "python":
            if '.insert(0,' in line or '.pop(0)' in line:
                patterns.append("inefficient_list_operations")
            if '+=' in line and 'str' in line.lower():
                patterns.append("string_concatenation")
        
        # Missing dictionary/map usage for lookups
        if re.search(r'for\s+\w+\s+in\s+\w+.*if\s+\w+\s*==', line):
            patterns.append("linear_search_instead_of_dict")
        
        return patterns
    
    def _is_string_concatenation(self, line: str, language: str) -> bool:
        """Check if line contains string concatenation"""
        if language == "python":
            return '+=' in line and ('str' in line.lower() or '"' in line or "'" in line)
        elif language in ["javascript", "typescript"]:
            return '+=' in line and ('"' in line or "'" in line or '`' in line)
        elif language == "java":
            return '+=' in line and ('String' in line or '"' in line)
        else:
            return '+=' in line and ('"' in line or "'" in line)
    
    def _is_synchronous_io(self, line: str, language: str) -> bool:
        """Check if line contains synchronous I/O operations"""
        sync_patterns = [
            # File operations
            r'\b(open|read|write|close)\(',
            # Network requests
            r'\b(requests\.get|requests\.post|urllib|fetch)\(',
            # Database operations without async
            r'\b(execute|query|commit)\(',
        ]
        
        # Check if it's not async
        if 'async' in line or 'await' in line:
            return False
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in sync_patterns)
    
    def _identify_io_operation(self, line: str) -> str:
        """Identify the type of I/O operation"""
        if re.search(r'\b(open|read|write|close)\(', line, re.IGNORECASE):
            return "FILE_IO"
        elif re.search(r'\b(requests|fetch|urllib)\b', line, re.IGNORECASE):
            return "NETWORK_REQUEST"
        elif re.search(r'\b(execute|query|commit)\(', line, re.IGNORECASE):
            return "DATABASE_IO"
        else:
            return "UNKNOWN_IO"
    
    def _enhance_prompt_with_performance_context(self, messages, context: ReviewContext, performance_insights: Dict):
        """Enhance prompt with performance-specific context"""
        # Get the human message (last message)
        human_message = messages[-1]
        
        # Add performance-specific instructions
        performance_context = self._build_performance_context(context, performance_insights)
        
        enhanced_content = f"{human_message.content}\n\n{performance_context}"
        
        # Replace the human message with enhanced version
        from langchain_core.messages import HumanMessage
        messages[-1] = HumanMessage(content=enhanced_content)
        
        return messages
    
    def _build_performance_context(self, context: ReviewContext, performance_insights: Dict) -> str:
        """Build performance-specific analysis context"""
        context_parts = []
        
        # Language-specific performance guidelines
        languages = set(f.language for f in context.file_changes if not f.is_binary)
        if languages:
            context_parts.append("**Language-Specific Performance Guidelines:**")
            
            for lang in languages:
                if lang == "python":
                    context_parts.append("- Python: Use list comprehensions, avoid string concatenation in loops, prefer sets for membership tests, use generators for large datasets")
                elif lang == "javascript":
                    context_parts.append("- JavaScript: Avoid blocking the event loop, use async/await, minimize DOM operations, use efficient array methods")
                elif lang == "java":
                    context_parts.append("- Java: Avoid autoboxing overhead, use appropriate collection types, consider parallel streams, optimize garbage collection")
                elif lang in ["cpp", "c"]:
                    context_parts.append("- C/C++: Use move semantics, avoid unnecessary copying, optimize memory access patterns, consider SIMD")
                elif lang == "go":
                    context_parts.append("- Go: Use buffered channels, avoid goroutine leaks, optimize memory allocations, use sync.Pool for reusable objects")
        
        # Performance insights from pre-analysis
        if any(performance_insights.values()):
            context_parts.append("\n**Pre-Analysis Performance Findings:**")
            
            if performance_insights["nested_loops"]:
                count = len(performance_insights["nested_loops"])
                context_parts.append(f"- Found {count} nested loop patterns (potential O(n²) complexity)")
            
            if performance_insights["database_queries_in_loops"]:
                count = len(performance_insights["database_queries_in_loops"])
                context_parts.append(f"- Found {count} database queries inside loops (N+1 query pattern)")
            
            if performance_insights["inefficient_data_structures"]:
                count = len(performance_insights["inefficient_data_structures"])
                context_parts.append(f"- Found {count} inefficient data structure usage patterns")
            
            if performance_insights["string_concatenation_in_loops"]:
                count = len(performance_insights["string_concatenation_in_loops"])
                context_parts.append(f"- Found {count} string concatenation operations in loops")
            
            if performance_insights["synchronous_io"]:
                count = len(performance_insights["synchronous_io"])
                context_parts.append(f"- Found {count} synchronous I/O operations that could be async")
        
        # Application context
        file_types = self._categorize_performance_context(context.file_changes)
        if file_types:
            context_parts.append(f"\n**Performance Context**: {', '.join(file_types)}")
        
        # Scale considerations
        total_changes = sum(len(f.additions) + len(f.deletions) for f in context.file_changes)
        if total_changes > 50:
            context_parts.append("\n**Scale**: Large changeset - focus on high-impact performance issues")
        else:
            context_parts.append("\n**Scale**: Small changeset - focus on critical performance bottlenecks")
        
        return "\n".join(context_parts)
    
    def _categorize_performance_context(self, file_changes) -> Set[str]:
        """Categorize files by performance context"""
        categories = set()
        
        for file_change in file_changes:
            path = file_change.file_path.lower()
            
            if any(keyword in path for keyword in ['api', 'service', 'controller']):
                categories.add("API/Service layer")
            elif any(keyword in path for keyword in ['database', 'db', 'repository', 'dao']):
                categories.add("Database layer")
            elif any(keyword in path for keyword in ['cache', 'redis', 'memcache']):
                categories.add("Caching layer")
            elif any(keyword in path for keyword in ['algorithm', 'compute', 'process']):
                categories.add("Computational logic")
            elif any(keyword in path for keyword in ['ui', 'frontend', 'client']):
                categories.add("User interface")
            elif any(keyword in path for keyword in ['test', 'spec']):
                categories.add("Test code")
            else:
                categories.add("Application logic")
        
        return categories
    
    def _validate_performance_findings(self, findings: List[Finding], context: ReviewContext) -> List[Finding]:
        """Validate and filter performance-specific findings"""
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
            
            # Validate description mentions performance impact
            description_lower = finding.description.lower()
            performance_keywords = [
                'performance', 'slow', 'inefficient', 'optimization', 'complexity',
                'memory', 'cpu', 'time', 'speed', 'scalability', 'bottleneck'
            ]
            
            if not any(keyword in description_lower for keyword in performance_keywords):
                logger.warning(f"Finding doesn't clearly indicate performance impact: {finding.description}")
                continue
            
            # Ensure suggestion is provided and mentions optimization
            if not finding.suggestion or len(finding.suggestion.strip()) < 15:
                logger.warning(f"Missing or insufficient optimization suggestion: {finding.suggestion}")
                continue
            
            suggestion_lower = finding.suggestion.lower()
            optimization_keywords = [
                'optimize', 'improve', 'cache', 'batch', 'async', 'parallel',
                'index', 'efficient', 'faster', 'reduce', 'avoid'
            ]
            
            if not any(keyword in suggestion_lower for keyword in optimization_keywords):
                logger.warning(f"Suggestion doesn't clearly indicate optimization: {finding.suggestion}")
                continue
            
            # Ensure category is correct
            finding.category = AnalysisCategory.PERFORMANCE
            finding.agent_source = self.agent_name
            
            validated.append(finding)
        
        return validated
    
    def get_performance_patterns(self) -> Dict[str, List[str]]:
        """Get performance patterns by language"""
        return {
            "python": [
                "List comprehensions over loops",
                "Sets for membership testing",
                "Generators for large datasets",
                "String join() over concatenation",
                "Dictionary lookups over linear search",
                "Caching expensive computations"
            ],
            "javascript": [
                "Async/await over callbacks",
                "Event delegation over individual handlers",
                "Debouncing/throttling for events",
                "Virtual DOM optimization",
                "Lazy loading and code splitting",
                "Web Workers for heavy computation"
            ],
            "java": [
                "Appropriate collection types",
                "Parallel streams for large datasets",
                "Connection pooling",
                "Avoiding autoboxing overhead",
                "StringBuilder for string building",
                "Caching and memoization"
            ],
            "cpp": [
                "Move semantics usage",
                "Memory access optimization",
                "SIMD instructions",
                "Cache-friendly data structures",
                "Avoiding unnecessary copying",
                "Template specialization"
            ],
            "go": [
                "Buffered channels",
                "Goroutine pooling",
                "Memory pool usage (sync.Pool)",
                "Efficient JSON parsing",
                "Avoiding memory allocations",
                "Profiling-guided optimization"
            ]
        }
    
    def get_agent_info(self) -> Dict[str, any]:
        """Get information about this agent"""
        info = super().get_agent_info()
        info.update({
            "specialization": "Performance optimization and efficiency",
            "analyzes": [
                "Algorithm efficiency",
                "Data structure choices",
                "Database query patterns",
                "Memory usage",
                "I/O operations",
                "Concurrency issues"
            ],
            "supported_patterns": list(self.get_performance_patterns().keys()),
            "requires_impact_explanation": True,
            "requires_optimization_suggestion": True
        })
        return info