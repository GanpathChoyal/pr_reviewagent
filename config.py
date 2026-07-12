import os


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


class Settings:
    github_token = os.getenv("GITHUB_TOKEN", "")

    llm_provider = os.getenv("LLM_PROVIDER", "GROQ")
    llm_model = os.getenv("LLM_MODEL", "GROQ-4")

    xai_api_key = os.getenv("XAI_API_KEY", "")
    GROQ_api_key = os.getenv("GROQ_API_KEY", xai_api_key)
    GROQ_logic_api_key = os.getenv("GROQ_LOGIC_API_KEY", GROQ_api_key)
    GROQ_readability_api_key = os.getenv("GROQ_READABILITY_API_KEY", GROQ_api_key)
    GROQ_performance_api_key = os.getenv("GROQ_PERFORMANCE_API_KEY", GROQ_api_key)
    GROQ_security_api_key = os.getenv("GROQ_SECURITY_API_KEY", GROQ_api_key)

    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

    # Add this line
    api_rate_limit = int(os.getenv("API_RATE_LIMIT", "60"))

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/pr_review_db",
    )

    persist_reviews = _env_bool("PERSIST_REVIEWS")


settings = Settings()