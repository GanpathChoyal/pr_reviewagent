"""
Comment Generator for formatting review findings into structured comments
"""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from models import Finding, ReviewSummary

logger = logging.getLogger(__name__)


class Comment:
    """Structured comment for a specific file and line"""
    
    def __init__(
        self,
        file_path: str,
        line_number: int,
        findings: List[Finding]
    ):
        self.file_path = file_path
        self.line_number = line_number
        self.findings = findings
    
    def to_dict(self) -> Dict:
        """Convert comment to dictionary"""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "agent_source": f.agent_source
                }
                for f in self.findings
            ]
        }


class FormattedReview:
    """Formatted review output"""
    
    def __init__(
        self,
        markdown_output: str,
        structured_comments: List[Comment],
        summary: ReviewSummary
    ):
        self.markdown_output = markdown_output
        self.structured_comments = structured_comments
        self.summary = summary


class CommentGenerator:
    """Generates formatted review comments from findings"""
    
    def __init__(self):
        """Initialize Comment Generator"""
        logger.debug("CommentGenerator initialized")
    
    def generate_comments(self, findings: List[Finding]) -> FormattedReview:
        """
        Generate formatted comments from findings
        
        Args:
            findings: List of findings from analyzer agents
            
        Returns:
            FormattedReview with markdown and structured comments
        """
        try:
            # Handle empty findings case
            if not findings:
                return self._generate_positive_review()
            
            # Group findings by file and line
            grouped_findings = self.group_by_file_and_line(findings)
            
            # Create structured comments
            structured_comments = self._create_structured_comments(grouped_findings)
            
            # Generate markdown output
            markdown_output = self.format_as_markdown(grouped_findings, findings)
            
            # Create summary
            summary = self._create_summary(findings)
            
            logger.info(
                f"Generated {len(structured_comments)} comments from {len(findings)} findings"
            )
            
            return FormattedReview(
                markdown_output=markdown_output,
                structured_comments=structured_comments,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Failed to generate comments: {e}", exc_info=True)
            # Return empty review on error
            return FormattedReview(
                markdown_output="# Review Failed\n\nAn error occurred while generating the review.",
                structured_comments=[],
                summary=ReviewSummary(
                    total_findings=0,
                    findings_by_severity={},
                    findings_by_category={},
                    files_analyzed=0,
                    lines_changed=0
                )
            )
    
    def group_by_file_and_line(
        self,
        findings: List[Finding]
    ) -> Dict[str, Dict[int, List[Finding]]]:
        """
        Group findings by file path and line number
        
        Args:
            findings: List of findings
            
        Returns:
            Nested dictionary: {file_path: {line_number: [findings]}}
        """
        grouped = defaultdict(lambda: defaultdict(list))
        
        for finding in findings:
            grouped[finding.file_path][finding.line_number].append(finding)
        
        logger.debug(f"Grouped {len(findings)} findings into {len(grouped)} files")
        
        return dict(grouped)
    
    def format_as_markdown(
        self,
        grouped_findings: Dict[str, Dict[int, List[Finding]]],
        all_findings: List[Finding]
    ) -> str:
        """
        Format findings as markdown
        
        Args:
            grouped_findings: Findings grouped by file and line
            all_findings: All findings for summary
            
        Returns:
            Markdown formatted string
        """
        lines = []
        
        # Header
        lines.append("# Code Review Results")
        lines.append("")
        
        # Summary
        summary = self._create_summary(all_findings)
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Issues Found:** {summary.total_findings}")
        lines.append(f"- **Files Analyzed:** {summary.files_analyzed}")
        lines.append("")
        
        # Severity breakdown
        if summary.findings_by_severity:
            lines.append("### By Severity")
            for severity, count in sorted(
                summary.findings_by_severity.items(),
                key=lambda x: self._severity_order(x[0]),
                reverse=True
            ):
                emoji = self._get_severity_emoji(severity)
                lines.append(f"- {emoji} **{severity.title()}:** {count}")
            lines.append("")
        
        # Category breakdown
        if summary.findings_by_category:
            lines.append("### By Category")
            for category, count in sorted(
                summary.findings_by_category.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                emoji = self._get_category_emoji(category)
                lines.append(f"- {emoji} **{category.title()}:** {count}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Findings by file
        lines.append("## Detailed Findings")
        lines.append("")
        
        for file_path in sorted(grouped_findings.keys()):
            lines.append(f"### 📄 `{file_path}`")
            lines.append("")
            
            line_findings = grouped_findings[file_path]
            
            for line_number in sorted(line_findings.keys()):
                findings_at_line = line_findings[line_number]
                
                # Line header
                lines.append(f"#### Line {line_number}")
                lines.append("")
                
                # Multiple findings on same line
                if len(findings_at_line) > 1:
                    lines.append(f"**{len(findings_at_line)} issues found on this line:**")
                    lines.append("")
                
                for i, finding in enumerate(findings_at_line, 1):
                    # Issue number if multiple
                    if len(findings_at_line) > 1:
                        lines.append(f"**Issue {i}:**")
                        lines.append("")
                    
                    # Severity and category badges
                    severity_badge = self._format_severity_badge(finding.severity.value)
                    category_badge = self._format_category_badge(finding.category.value)
                    lines.append(f"{severity_badge} {category_badge}")
                    lines.append("")
                    
                    # Description
                    lines.append(f"**Description:** {finding.description}")
                    lines.append("")
                    
                    # Suggestion if available
                    if finding.suggestion:
                        lines.append(f"**Suggestion:** {finding.suggestion}")
                        lines.append("")
                    
                    # Agent source
                    lines.append(f"*Detected by: {finding.agent_source}*")
                    lines.append("")
                    
                    # Separator between issues
                    if i < len(findings_at_line):
                        lines.append("---")
                        lines.append("")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_structured_comments(
        self,
        grouped_findings: Dict[str, Dict[int, List[Finding]]]
    ) -> List[Comment]:
        """Create structured comment objects"""
        comments = []
        
        for file_path, line_findings in grouped_findings.items():
            for line_number, findings in line_findings.items():
                comment = Comment(
                    file_path=file_path,
                    line_number=line_number,
                    findings=findings
                )
                comments.append(comment)
        
        return comments
    
    def _create_summary(self, findings: List[Finding]) -> ReviewSummary:
        """Create review summary from findings"""
        # Count by severity
        findings_by_severity = defaultdict(int)
        for finding in findings:
            findings_by_severity[finding.severity.value] += 1
        
        # Count by category
        findings_by_category = defaultdict(int)
        for finding in findings:
            findings_by_category[finding.category.value] += 1
        
        # Count unique files
        files_analyzed = len(set(f.file_path for f in findings))
        
        return ReviewSummary(
            total_findings=len(findings),
            findings_by_severity=dict(findings_by_severity),
            findings_by_category=dict(findings_by_category),
            files_analyzed=files_analyzed,
            lines_changed=0  # This would need to be passed from diff parser
        )
    
    def _generate_positive_review(self) -> FormattedReview:
        """Generate positive review when no issues found"""
        markdown = """# Code Review Results

## ✅ No Issues Found!

Great work! The code review didn't identify any issues in the following areas:

- 🔍 **Logic:** No logical errors or bugs detected
- 📖 **Readability:** Code is clear and maintainable
- ⚡ **Performance:** No performance concerns identified
- 🔒 **Security:** No security vulnerabilities found

Keep up the excellent work!
"""
        
        summary = ReviewSummary(
            total_findings=0,
            findings_by_severity={},
            findings_by_category={},
            files_analyzed=0,
            lines_changed=0
        )
        
        return FormattedReview(
            markdown_output=markdown,
            structured_comments=[],
            summary=summary
        )
    
    def _format_severity_badge(self, severity: str) -> str:
        """Format severity as a badge"""
        severity_colors = {
            "critical": "🔴 **CRITICAL**",
            "high": "🟠 **HIGH**",
            "medium": "🟡 **MEDIUM**",
            "low": "🟢 **LOW**"
        }
        return severity_colors.get(severity.lower(), f"**{severity.upper()}**")
    
    def _format_category_badge(self, category: str) -> str:
        """Format category as a badge"""
        category_labels = {
            "logic": "🐛 Logic",
            "readability": "📖 Readability",
            "performance": "⚡ Performance",
            "security": "🔒 Security"
        }
        label = category_labels.get(category.lower(), category.title())
        return f"`{label}`"
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity"""
        emojis = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }
        return emojis.get(severity.lower(), "⚪")
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emojis = {
            "logic": "🐛",
            "readability": "📖",
            "performance": "⚡",
            "security": "🔒"
        }
        return emojis.get(category.lower(), "📝")
    
    def _severity_order(self, severity: str) -> int:
        """Get numeric order for severity"""
        order = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return order.get(severity.lower(), 0)