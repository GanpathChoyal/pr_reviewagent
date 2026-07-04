"""
Streamlit Frontend for PR Review Agent
"""
import streamlit as st
import requests
from datetime import datetime
import json

# Configuration
API_BASE_URL = "https://pr-reviewagent.onrender.com"

# Page config
st.set_page_config(
    page_title="PR Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
   .finding-card {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
    color: black;
}
    .critical { border-left-color: #d32f2f; background-color: #ffebee; }
    .high { border-left-color: #f57c00; background-color: #fff3e0; }
    .medium { border-left-color: #fbc02d; background-color: #fffde7; }
    .low { border-left-color: #388e3c; background-color: #e8f5e9; }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def submit_review(pr_url=None, diff_content=None, repository=None, pr_number=None, 
                  github_token=None, config=None):
    """Submit review request to API"""
    payload = {
        "pr_url": pr_url,
        "diff_content": diff_content,
        "repository": repository,
        "pr_number": pr_number,
        "github_token": github_token,
        "config": config
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/reviews",
            json=payload,
            timeout=600  # 10 minutes for large PRs
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        # Try to get detailed error message from response
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        return None, f"API Error: {error_detail}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The review is taking too long. Try with a smaller PR."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure the backend is running on http://localhost:8000"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def get_review_history(filters=None):
    """Get review history from API"""
    try:
        params = filters or {}
        response = requests.get(
            f"{API_BASE_URL}/api/reviews/history",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


def display_finding(finding, index):
    """Display a single finding"""
    severity = finding.get('severity', 'low').lower()
    category = finding.get('category', 'unknown')
    
    with st.container():
        st.markdown(f"""
            <div class="finding-card {severity}">
                <h4>🔍 Finding #{index + 1} - {severity.upper()}</h4>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>File:</strong> {finding.get('file_path', 'N/A')} (Line {finding.get('line_number', 'N/A')})</p>
                <p><strong>Issue:</strong> {finding.get('message', 'No message')}</p>
                <p><strong>Suggestion:</strong> {finding.get('suggestion', 'No suggestion')}</p>
            </div>
        """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">🔍 PR Review Agent</p>', unsafe_allow_html=True)
    st.markdown("Automated code review powered by AI agents")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the backend server.")
        st.code("python main.py", language="bash")
        return
    
    st.success("✅ API is running")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Review mode selection
        review_mode = st.radio(
            "Review Mode",
            ["GitHub PR URL", "Manual Diff", "View History"],
            help="Choose how you want to submit code for review"
        )
        
        st.divider()
        
        # Analysis configuration
        st.subheader("Analysis Settings")
        
        severity_threshold = st.selectbox(
            "Severity Threshold",
            ["low", "medium", "high", "critical"],
            index=0,
            help="Minimum severity level to report"
        )
        
        enabled_categories = st.multiselect(
            "Analysis Categories",
            ["logic", "security", "performance", "readability"],
            default=["logic", "security", "performance", "readability"],
            help="Select which types of issues to analyze"
        )
        
        custom_rules = st.text_area(
            "Custom Rules (optional)",
            placeholder="Enter custom review rules...",
            help="Add specific rules for the review"
        )
    
    # Main content area
    if review_mode == "GitHub PR URL":
        st.header("📋 Review GitHub Pull Request")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            pr_url = st.text_input(
                "GitHub PR URL",
                placeholder="https://github.com/owner/repo/pull/123",
                help="Enter the full URL of the GitHub pull request"
            )
        
        with col2:
            github_token = st.text_input(
                "GitHub Token (optional)",
                type="password",
                help="Personal access token for private repos. Leave empty to use token from .env file"
            )
        
        st.info("💡 **Tip:** For large repos like PyTorch, make sure you have a valid GitHub token configured. See GITHUB_TOKEN_SETUP.md for instructions.")
        
        if st.button("🚀 Start Review", type="primary", use_container_width=True):
            if not pr_url:
                st.error("Please enter a PR URL")
            else:
                with st.spinner("Analyzing pull request... This may take a few minutes."):
                    config = {
                        "severity_threshold": severity_threshold,
                        "enabled_categories": enabled_categories,
                        "custom_rules": custom_rules if custom_rules else None
                    }
                    
                    result, error = submit_review(
                        pr_url=pr_url,
                        github_token=github_token if github_token else None,
                        config=config
                    )
                    
                    if error:
                        st.error(f"❌ Review failed: {error}")
                    else:
                        st.success("✅ Review completed!")
                        display_review_results(result)
    
    elif review_mode == "Manual Diff":
        st.header("📝 Review Manual Diff")
        
        col1, col2 = st.columns(2)
        
        with col1:
            repository = st.text_input(
                "Repository Name",
                placeholder="owner/repo",
                help="Repository identifier"
            )
        
        with col2:
            pr_number = st.number_input(
                "PR Number (optional)",
                min_value=0,
                value=0,
                help="Pull request number if applicable"
            )
        
        diff_content = st.text_area(
            "Diff Content",
            height=300,
            placeholder="Paste your git diff here...",
            help="Paste the output of 'git diff' command"
        )
        
        if st.button("🚀 Start Review", type="primary", use_container_width=True):
            if not diff_content:
                st.error("Please enter diff content")
            else:
                with st.spinner("Analyzing diff... This may take a few minutes."):
                    config = {
                        "severity_threshold": severity_threshold,
                        "enabled_categories": enabled_categories,
                        "custom_rules": custom_rules if custom_rules else None
                    }
                    
                    result, error = submit_review(
                        diff_content=diff_content,
                        repository=repository if repository else None,
                        pr_number=pr_number if pr_number > 0 else None,
                        config=config
                    )
                    
                    if error:
                        st.error(f"❌ Review failed: {error}")
                    else:
                        st.success("✅ Review completed!")
                        display_review_results(result)
    
    else:  # View History
        st.header("📚 Review History")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_repo = st.text_input("Filter by Repository", placeholder="owner/repo")
        
        with col2:
            filter_pr = st.number_input("Filter by PR Number", min_value=0, value=0)
        
        with col3:
            limit = st.number_input("Results Limit", min_value=1, max_value=100, value=10)
        
        if st.button("🔍 Load History", use_container_width=True):
            filters = {
                "limit": limit
            }
            if filter_repo:
                filters["repository"] = filter_repo
            if filter_pr > 0:
                filters["pr_number"] = filter_pr
            
            with st.spinner("Loading review history..."):
                history, error = get_review_history(filters)
                
                if error:
                    st.error(f"❌ Failed to load history: {error}")
                else:
                    if not history:
                        st.info("No reviews found")
                    else:
                        st.success(f"Found {len(history)} review(s)")
                        for review in history:
                            display_history_item(review)


def display_review_results(result):
    """Display review results"""
    st.divider()
    
    # Summary
    summary = result.get('summary', {})
    findings = result.get('findings', [])
    pr_metadata = result.get('pr_metadata', {})
    
    # Metadata
    with st.expander("📊 Review Metadata", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Repository", pr_metadata.get('repository', 'N/A'))
            st.metric("PR Number", pr_metadata.get('pr_number', 'N/A'))
        
        with col2:
            st.metric("Total Findings", summary.get('total_findings', 0))
            st.metric("Files Analyzed", summary.get('files_analyzed', 0))
        
        with col3:
            st.metric("Lines Changed", summary.get('lines_changed', 0))
            timestamp = result.get('timestamp', '')
            if timestamp:
                st.metric("Timestamp", timestamp.split('T')[0])
    
    # Findings by severity
    if summary.get('findings_by_severity'):
        st.subheader("📈 Findings by Severity")
        severity_data = summary['findings_by_severity']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Critical", severity_data.get('critical', 0))
        with col2:
            st.metric("🟠 High", severity_data.get('high', 0))
        with col3:
            st.metric("🟡 Medium", severity_data.get('medium', 0))
        with col4:
            st.metric("🟢 Low", severity_data.get('low', 0))
    
    # Findings by category
    if summary.get('findings_by_category'):
        st.subheader("📊 Findings by Category")
        category_data = summary['findings_by_category']
        
        cols = st.columns(len(category_data))
        for idx, (category, count) in enumerate(category_data.items()):
            with cols[idx]:
                st.metric(category.title(), count)
    
    # Individual findings
    if findings:
        st.divider()
        st.subheader(f"🔍 Detailed Findings ({len(findings)})")
        
        # Filter findings by severity
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["critical", "high", "medium", "low"],
            default=["critical", "high", "medium", "low"]
        )
        
        filtered_findings = [
            f for f in findings 
            if f.get('severity', 'low').lower() in severity_filter
        ]
        
        for idx, finding in enumerate(filtered_findings):
            display_finding(finding, idx)
    else:
        st.info("🎉 No issues found! Your code looks great!")
    
    # Download results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(result, indent=2),
            file_name=f"review_{result.get('review_id', 'unknown')}.json",
            mime="application/json"
        )
    
    with col2:
        formatted_comments = result.get('formatted_comments', '')
        if formatted_comments:
            st.download_button(
                label="📥 Download Comments",
                data=formatted_comments,
                file_name=f"comments_{result.get('review_id', 'unknown')}.txt",
                mime="text/plain"
            )


def display_history_item(review):
    """Display a single history item"""
    pr_metadata = review.get('pr_metadata', {})
    findings = review.get('findings', [])
    timestamp = review.get('timestamp', '')
    
    with st.expander(
        f"📋 {pr_metadata.get('repository', 'Unknown')} - PR #{pr_metadata.get('pr_number', 'N/A')} - {len(findings)} findings",
        expanded=False
    ):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Review ID:** {review.get('review_id', 'N/A')}")
            st.write(f"**Repository:** {pr_metadata.get('repository', 'N/A')}")
            st.write(f"**PR Number:** {pr_metadata.get('pr_number', 'N/A')}")
        
        with col2:
            st.write(f"**Timestamp:** {timestamp}")
            st.write(f"**Total Findings:** {len(findings)}")
        
        if findings:
            st.write("**Findings:**")
            for idx, finding in enumerate(findings[:5]):  # Show first 5
                display_finding(finding, idx)
            
            if len(findings) > 5:
                st.info(f"... and {len(findings) - 5} more findings")


if __name__ == "__main__":
    main()