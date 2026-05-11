"""Dependency injection module for FastAPI."""
from fastapi import HTTPException

from src.config import Config


def get_hf_token() -> str:
    """Get HuggingFace token from config or environment."""
    token = Config.HF_TOKEN
    if not token:
        raise HTTPException(
            status_code=500,
            detail=(
                "HF_TOKEN is not configured. "
                "Set it in environment or .env file."
            ),
        )
    return token


def get_config() -> Config:
    """Get the application configuration."""
    return Config