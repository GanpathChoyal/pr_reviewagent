import os


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


class Settings:
    github_token = os.getenv("GITHUB_TOKEN", "")

    llm_provider = os.getenv("LLM_PROVIDER", "gemini")
    llm_model = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", google_api_key)
    gemini_logic_api_key = os.getenv("GEMINI_LOGIC_API_KEY", gemini_api_key)
    gemini_readability_api_key = os.getenv("GEMINI_READABILITY_API_KEY", gemini_api_key)
    gemini_performance_api_key = os.getenv("GEMINI_PERFORMANCE_API_KEY", gemini_api_key)
    gemini_security_api_key = os.getenv("GEMINI_SECURITY_API_KEY", gemini_api_key)

    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

    # Add this line
    api_rate_limit = int(os.getenv("API_RATE_LIMIT", "60"))

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/pr_review_db",
    )

    persist_reviews = _env_bool("PERSIST_REVIEWS")


settings = Settings()