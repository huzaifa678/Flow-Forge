"""Root FastAPI application for FlowForge."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Config
from src.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifecycle manager."""
    # Startup: validate configuration
    if not Config.HF_TOKEN:
        import warnings

        warnings.warn(
            "HF_TOKEN is not set. API endpoints that require HuggingFace "
            "will fail. Set HF_TOKEN in your environment or .env file.",
            UserWarning,
        )
    yield
    # Shutdown cleanup if needed
    pass


app = FastAPI(
    title="FlowForge API",
    description=(
        "FlowForge is an AI-powered project planning and diagram generation platform. "
        "It uses a multi-agent pipeline to generate project timelines, plans, "
        "and technical diagrams from client proposals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)