"""
Security Analyzer Agent for detecting security vulnerabilities
"""
import logging
import re
from typing import List, Dict, Set

from langchain_core.language_models import BaseLanguageModel

from agents.base_agent import AnalyzerAgent, AgentConfig
from models import Finding, ReviewContext, AnalysisCategory

logger = logging.getLogger(__name__)


class SecurityAnalyzerAgent(AnalyzerAgent):
    """Analyzer agent specialized in detecting security vulnerabilities"""

    def __init__(self, llm: BaseLanguageModel, config: AgentConfig = None):
        """
        Initialize Security Analyzer Agent

        Args:
            llm: Language model for analysis
            config: Agent configuration
        """
        super().__init__(llm, config)
        self.agent_name = "security_analyzer"

    def get_analysis_category(self) -> AnalysisCategory:
        """Get the analysis category for this agent"""
        return AnalysisCategory.SECURITY

    def get_system_prompt(self) -> str:
        """Get the system prompt for security analysis"""
        return """You are an expert security code reviewer specializing in identifying security vulnerabilities and weaknesses. Your task is to analyze code changes and identify potential security issues that could lead to data breaches, unauthorized access, or other security incidents.

Focus on detecting these types of security vulnerabilities:

1. **SQL Injection Vulnerabilities**:
   - Direct string concatenation in SQL queries
   - Unsanitized user input in database queries
   - Missing parameterized queries or prepared statements
   - Dynamic SQL construction without proper escaping
   - ORM queries vulnerable to injection

2. **Cross-Site Scripting (XSS)**:
   - Unescaped user input rendered in HTML
   - Direct DOM manipulation with user data
   - Missing output encoding
   - Unsafe use of innerHTML or dangerouslySetInnerHTML
   - Reflected or stored XSS vulnerabilities

3. **Authentication and Authorization Issues**:
   - Weak password requirements or storage
   - Missing authentication checks
   - Hardcoded credentials or API keys
   - Insecure session management
   - Missing authorization checks for sensitive operations
   - Broken access control
   - JWT token vulnerabilities

4. **Sensitive Data Exposure**:
   - Logging sensitive information (passwords, tokens, PII)
   - Exposing sensitive data in error messages
   - Unencrypted sensitive data transmission
   - Missing data encryption at rest
   - Sensitive data in URLs or query parameters
   - Information disclosure through verbose errors

5. **Input Validation and Sanitization**:
   - Missing input validation
   - Insufficient input sanitization
   - Type confusion vulnerabilities
   - Path traversal vulnerabilities
   - Command injection risks
   - XML/JSON injection

6. **Cryptographic Issues**:
   - Use of weak or deprecated cryptographic algorithms
   - Hardcoded encryption keys
   - Insufficient randomness in security-critical operations
   - Missing or improper certificate validation
   - Insecure random number generation

7. **Security Misconfigurations**:
   - Debug mode enabled in production
   - Overly permissive CORS policies
   - Missing security headers
   - Insecure default configurations
   - Exposed administrative interfaces

8. **Dependency and Third-Party Risks**:
   - Use of known vulnerable dependencies
   - Insecure deserialization
   - Unsafe use of eval() or similar functions
   - Loading untrusted code or resources

9. **API Security Issues**:
   - Missing rate limiting
   - Insufficient API authentication
   - Mass assignment vulnerabilities
   - Insecure direct object references (IDOR)
   - Missing CSRF protection

10. **Code Injection Vulnerabilities**:
    - Command injection
    - Code injection through eval()
    - Template injection
    - LDAP injection
    - XML External Entity (XXE) injection

**Analysis Guidelines**:
- Prioritize vulnerabilities by potential impact and exploitability
- Consider the context and environment (web app, API, mobile, etc.)
- Provide specific remediation guidance with code examples when possible
- Reference security standards (OWASP Top 10, CWE) when applicable
- Be thorough but avoid false positives
- Consider defense-in-depth principles

**Severity Levels**:
- **Critical**: Immediate exploitation possible, high impact (data breach, system compromise)
- **High**: Exploitable vulnerability with significant impact
- **Medium**: Vulnerability requiring specific conditions or moderate impact
- **Low**: Minor security weakness or best practice violation

**Output Format**:
Return findings as a JSON array. Each finding MUST include severity level and remediation guidance:

```json
[
  {
    "file_path": "src/api/user_controller.py",
    "line_number": 42,
    "severity": "critical",
    "description": "SQL Injection vulnerability: User input 'username' is directly concatenated into SQL query without sanitization or parameterization",
    "suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE username = ?', (username,)) or use ORM methods like User.objects.filter(username=username)"
  }
]
```

**Important**: Every finding MUST include:
- Clear description of the security vulnerability
- Severity level based on exploitability and impact
- Specific remediation guidance with secure code examples
- Reference to security standards when applicable (e.g., "OWASP A1: Injection")

Return an empty array [] if no security vulnerabilities are found."""

    async def analyze(self, context: ReviewContext) -> List[Finding]:
        """
        Analyze code changes for security vulnerabilities

        Args:
            context: Review context with file changes and configuration

        Returns:
            List of findings related to security issues
        """
        # Validate context
        if not self.validate_context(context):
            logger.debug("Context validation failed for security analysis")
            return []

        try:
            # Pre-analyze code for security patterns
            security_insights = self._analyze_security_patterns(context)

            # Create prompt messages
            messages = self.create_prompt(context)

            # Add security-specific context
            enhanced_messages = self._enhance_prompt_with_security_context(
                messages, context, security_insights
            )

            # Invoke LLM with retry
            response = await self._invoke_llm_with_retry(enhanced_messages)

            # Parse response into findings
            findings = await self.parse_llm_response(response)

            # Filter and validate findings
            validated_findings = self._validate_security_findings(findings, context)

            logger.info(f"Security analyzer found {len(validated_findings)} vulnerabilities")
            return validated_findings

        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return []

    def _analyze_security_patterns(self, context: ReviewContext) -> Dict[str, any]:
        """Pre-analyze code for security anti-patterns"""
        insights = {
            "sql_injection_risks": [],
            "xss_risks": [],
            "hardcoded_secrets": [],
            "weak_crypto": [],
            "input_validation_missing": [],
            "sensitive_data_exposure": [],
            "authentication_issues": [],
            "insecure_deserialization": []
        }

        for file_change in context.file_changes:
            if file_change.is_binary:
                continue

            # Analyze additions for security patterns
            for line_change in file_change.additions:
                line_content = line_change.content.strip()

                if not line_content:
                    continue

                # Check for SQL injection risks
                if self._has_sql_injection_risk(line_content, file_change.language):
                    insights["sql_injection_risks"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "pattern": "string_concatenation_in_query"
                    })

                # Check for XSS risks
                if self._has_xss_risk(line_content, file_change.language):
                    insights["xss_risks"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "pattern": "unsafe_html_rendering"
                    })

                # Check for hardcoded secrets
                secrets = self._find_hardcoded_secrets(line_content)
                for secret_type in secrets:
                    insights["hardcoded_secrets"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "type": secret_type
                    })

                # Check for weak cryptography
                if self._has_weak_crypto(line_content, file_change.language):
                    insights["weak_crypto"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number
                    })

                # Check for sensitive data in logs
                if self._logs_sensitive_data(line_content, file_change.language):
                    insights["sensitive_data_exposure"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number,
                        "type": "logging"
                    })

                # Check for insecure deserialization
                if self._has_insecure_deserialization(line_content, file_change.language):
                    insights["insecure_deserialization"].append({
                        "file": file_change.file_path,
                        "line": line_change.line_number
                    })

        return insights

    def _has_sql_injection_risk(self, line: str, language: str) -> bool:
        """Check if line has SQL injection risk"""
        # String concatenation with SQL keywords
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE']

        # Check for string concatenation with SQL
        if any(keyword in line.upper() for keyword in sql_keywords):
            # Look for string concatenation patterns
            if language == "python":
                if any(pattern in line for pattern in [' + ', '.format(', 'f"', "f'"]):
                    return True
            elif language in ["javascript", "typescript"]:
                if any(pattern in line for pattern in [' + ', '${', '`']):
                    return True
            elif language == "java":
                if ' + ' in line or '.concat(' in line:
                    return True

        # Check for execute with string formatting
        if re.search(r'\.execute\s*\(\s*["\'].*[+%]', line):
            return True

        return False

    def _has_xss_risk(self, line: str, language: str) -> bool:
        """Check if line has XSS risk"""
        xss_patterns = [
            r'\.innerHTML\s*=',
            r'dangerouslySetInnerHTML',
            r'document\.write\(',
            r'\.html\(',  # jQuery
            r'v-html=',  # Vue.js
            r'\[innerHTML\]',  # Angular
        ]

        return any(re.search(pattern, line, re.IGNORECASE) for pattern in xss_patterns)

    def _find_hardcoded_secrets(self, line: str) -> List[str]:
        """Find hardcoded secrets in code"""
        secrets = []

        # Common secret patterns
        secret_patterns = {
            'password': r'password\s*=\s*["\'][^"\']{3,}["\']',
            'api_key': r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
            'secret': r'secret\s*=\s*["\'][^"\']{10,}["\']',
            'token': r'token\s*=\s*["\'][^"\']{10,}["\']',
            'private_key': r'private[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
            'aws_key': r'aws[_-]?(access[_-]?)?key[_-]?id\s*=\s*["\'][^"\']{10,}["\']',
        }

        for secret_type, pattern in secret_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                # Exclude common test/placeholder values
                if not any(placeholder in line.lower() for placeholder in [
                    'your_', 'example', 'test', 'dummy', 'fake', 'placeholder', 'xxx'
                ]):
                    secrets.append(secret_type)

        return secrets

    def _has_weak_crypto(self, line: str, language: str) -> bool:
        """Check for weak cryptographic algorithms"""
        weak_crypto_patterns = [
            r'\bMD5\b',
            r'\bSHA1\b',
            r'\bDES\b',
            r'\bRC4\b',
            r'\.md5\(',
            r'\.sha1\(',
            r'hashlib\.md5',
            r'hashlib\.sha1',
            r'crypto\.createHash\(["\']md5',
            r'crypto\.createHash\(["\']sha1',
        ]

        return any(re.search(pattern, line, re.IGNORECASE) for pattern in weak_crypto_patterns)

    def _logs_sensitive_data(self, line: str, language: str) -> bool:
        """Check if line logs sensitive data"""
        # Check if it's a logging statement
        logging_patterns = [
            r'log\.',
            r'logger\.',
            r'console\.',
            r'print\(',
            r'println\(',
            r'System\.out',
        ]

        is_logging = any(re.search(pattern, line, re.IGNORECASE) for pattern in logging_patterns)

        if not is_logging:
            return False

        # Check for sensitive data keywords
        sensitive_keywords = [
            'password', 'passwd', 'pwd',
            'token', 'secret', 'key',
            'credit_card', 'creditcard', 'ssn',
            'api_key', 'apikey',
            'private_key', 'privatekey',
            'authorization', 'auth',
        ]

        return any(keyword in line.lower() for keyword in sensitive_keywords)

    def _has_insecure_deserialization(self, line: str, language: str) -> bool:
        """Check for insecure deserialization"""
        if language == "python":
            return 'pickle.loads' in line or 'yaml.load(' in line
        elif language in ["javascript", "typescript"]:
            return 'eval(' in line or 'Function(' in line
        elif language == "java":
            return 'readObject()' in line or 'XMLDecoder' in line

        return False

    def _enhance_prompt_with_security_context(
        self, messages, context: ReviewContext, security_insights: Dict
    ):
        """Enhance prompt with security-specific context"""
        # Get the human message (last message)
        human_message = messages[-1]

        # Add security-specific instructions
        security_context = self._build_security_context(context, security_insights)

        enhanced_content = f"{human_message.content}\n\n{security_context}"

        # Replace the human message with enhanced version
        from langchain_core.messages import HumanMessage
        messages[-1] = HumanMessage(content=enhanced_content)

        return messages

    def _build_security_context(self, context: ReviewContext, security_insights: Dict) -> str:
        """Build security-specific analysis context"""
        context_parts = []

        # Language-specific security guidelines
        languages = set(f.language for f in context.file_changes if not f.is_binary)
        if languages:
            context_parts.append("**Language-Specific Security Guidelines:**")

            for lang in languages:
                if lang == "python":
                    context_parts.append("- Python: Watch for pickle deserialization, SQL injection in raw queries, command injection in subprocess, YAML unsafe loading")
                elif lang == "javascript":
                    context_parts.append("- JavaScript: Check for XSS via innerHTML, eval() usage, prototype pollution, insecure dependencies")
                elif lang == "java":
                    context_parts.append("- Java: Look for SQL injection, XXE vulnerabilities, insecure deserialization, path traversal")
                elif lang in ["cpp", "c"]:
                    context_parts.append("- C/C++: Check for buffer overflows, format string vulnerabilities, use-after-free, integer overflows")
                elif lang == "go":
                    context_parts.append("- Go: Watch for SQL injection, command injection, path traversal, insecure TLS configuration")

        # Security insights from pre-analysis
        if any(security_insights.values()):
            context_parts.append("\n**Pre-Analysis Security Findings:**")

            if security_insights["sql_injection_risks"]:
                count = len(security_insights["sql_injection_risks"])
                context_parts.append(f"- Found {count} potential SQL injection risks")

            if security_insights["xss_risks"]:
                count = len(security_insights["xss_risks"])
                context_parts.append(f"- Found {count} potential XSS vulnerabilities")

            if security_insights["hardcoded_secrets"]:
                count = len(security_insights["hardcoded_secrets"])
                context_parts.append(f"- Found {count} potential hardcoded secrets")

            if security_insights["weak_crypto"]:
                count = len(security_insights["weak_crypto"])
                context_parts.append(f"- Found {count} uses of weak cryptographic algorithms")

            if security_insights["sensitive_data_exposure"]:
                count = len(security_insights["sensitive_data_exposure"])
                context_parts.append(f"- Found {count} potential sensitive data exposures")

            if security_insights["insecure_deserialization"]:
                count = len(security_insights["insecure_deserialization"])
                context_parts.append(f"- Found {count} insecure deserialization patterns")

        # Application context
        file_types = self._categorize_security_context(context.file_changes)
        if file_types:
            context_parts.append(f"\n**Security Context**: {', '.join(file_types)}")

        # OWASP Top 10 reminder
        context_parts.append("\n**OWASP Top 10 Focus Areas**: Injection, Broken Authentication, Sensitive Data Exposure, XXE, Broken Access Control, Security Misconfiguration, XSS, Insecure Deserialization, Using Components with Known Vulnerabilities, Insufficient Logging & Monitoring")

        return "\n".join(context_parts)

    def _categorize_security_context(self, file_changes) -> Set[str]:
        """Categorize files by security context"""
        categories = set()

        for file_change in file_changes:
            path = file_change.file_path.lower()

            if any(keyword in path for keyword in ['auth', 'login', 'session']):
                categories.add("Authentication/Authorization")
            elif any(keyword in path for keyword in ['api', 'controller', 'endpoint']):
                categories.add("API endpoints")
            elif any(keyword in path for keyword in ['database', 'db', 'sql', 'query']):
                categories.add("Database layer")
            elif any(keyword in path for keyword in ['crypto', 'encrypt', 'hash']):
                categories.add("Cryptography")
            elif any(keyword in path for keyword in ['user', 'input', 'form']):
                categories.add("User input handling")
            elif any(keyword in path for keyword in ['config', 'setting']):
                categories.add("Configuration")
            elif any(keyword in path for keyword in ['payment', 'billing']):
                categories.add("Payment processing")
            else:
                categories.add("Application logic")

        return categories

    def _validate_security_findings(self, findings: List[Finding], context: ReviewContext) -> List[Finding]:
        """Validate and filter security-specific findings"""
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

            # Validate description mentions security
            description_lower = finding.description.lower()
            security_keywords = [
                'security', 'vulnerability', 'injection', 'xss', 'authentication',
                'authorization', 'credential', 'password', 'token', 'exploit',
                'attack', 'malicious', 'unsafe', 'insecure', 'exposure'
            ]

            if not any(keyword in description_lower for keyword in security_keywords):
                logger.warning(f"Finding doesn't clearly indicate security issue: {finding.description}")
                continue

            # Ensure suggestion/remediation is provided
            if not finding.suggestion or len(finding.suggestion.strip()) < 20:
                logger.warning(f"Missing or insufficient remediation guidance: {finding.suggestion}")
                continue

            suggestion_lower = finding.suggestion.lower()
            remediation_keywords = [
                'use', 'implement', 'add', 'validate', 'sanitize', 'escape',
                'encrypt', 'hash', 'parameterize', 'secure', 'protect', 'fix'
            ]

            if not any(keyword in suggestion_lower for keyword in remediation_keywords):
                logger.warning(f"Suggestion doesn't provide clear remediation: {finding.suggestion}")
                continue

            # Ensure category is correct
            finding.category = AnalysisCategory.SECURITY
            finding.agent_source = self.agent_name

            validated.append(finding)

        return validated

    def get_security_patterns(self) -> Dict[str, List[str]]:
        """Get security vulnerability patterns by category"""
        return {
            "injection": [
                "SQL injection",
                "Command injection",
                "LDAP injection",
                "XPath injection",
                "XML injection",
                "Code injection"
            ],
            "authentication": [
                "Weak password requirements",
                "Hardcoded credentials",
                "Insecure session management",
                "Missing authentication",
                "Broken access control"
            ],
            "data_exposure": [
                "Sensitive data in logs",
                "Unencrypted data transmission",
                "Information disclosure",
                "Verbose error messages",
                "PII exposure"
            ],
            "cryptography": [
                "Weak algorithms (MD5, SHA1, DES)",
                "Hardcoded encryption keys",
                "Insufficient randomness",
                "Missing certificate validation"
            ],
            "xss": [
                "Reflected XSS",
                "Stored XSS",
                "DOM-based XSS",
                "Unsafe HTML rendering"
            ],
            "misconfiguration": [
                "Debug mode in production",
                "Permissive CORS",
                "Missing security headers",
                "Default credentials"
            ]
        }

    def get_agent_info(self) -> Dict[str, any]:
        """Get information about this agent"""
        info = super().get_agent_info()
        info.update({
            "specialization": "Security vulnerability detection",
            "detects": [
                "SQL injection",
                "XSS vulnerabilities",
                "Authentication issues",
                "Sensitive data exposure",
                "Weak cryptography",
                "Input validation gaps",
                "Security misconfigurations"
            ],
            "standards": ["OWASP Top 10", "CWE"],
            "requires_severity_assignment": True,
            "requires_remediation_guidance": True
        })
        return info