"""
Configuration - loads settings from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# --- Slack ---
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
MY_SLACK_USER_ID = os.getenv("MY_SLACK_USER_ID")

# --- Anthropic (Claude) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# --- OpenAI (embeddings only) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAG Settings ---
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "15"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# --- Retention & Recency ---
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "60"))
RECENCY_WEIGHT = float(os.getenv("RECENCY_WEIGHT", "0.3"))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def validate():
    """Check that all required settings are present."""
    missing = []
    for name in [
        "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_SIGNING_SECRET",
        "MY_SLACK_USER_ID", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    ]:
        if not globals().get(name):
            missing.append(name)
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your values."
        )
