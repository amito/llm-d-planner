"""Health check endpoint."""

import logging

from fastapi import APIRouter

from planner.llm.factory import create_llm_client

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "planner"}


@router.get("/api/v1/llm-status")
async def llm_status():
    """Check whether the configured LLM provider is reachable."""
    try:
        client = create_llm_client()
        available = client.is_available()
    except Exception as e:
        logger.warning("LLM status check failed: %s", e)
        available = False
    return {"available": available}
