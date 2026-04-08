"""
server/app.py — FastAPI application exposing the OpenEnv HTTP API.

Endpoints (OpenEnv spec):
  GET  /           → health / ping (returns 200 + environment info)
  POST /reset      → reset the environment, start a new episode
  POST /step       → submit an action, get observation + reward + done
  GET  /state      → return full internal state (for grading / debug)
  GET  /health     → lightweight health check
  GET  /schema     → return JSON schemas for action, observation, state
  GET  /tasks      → list available tasks with descriptions
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the import path so we can import models/environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    HealthResponse,
    NegotiationAction,
    NegotiationObservation,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)
from environment import NegotiationEnvironment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("negotiation_env")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Negotiation Environment",
    description=(
        "An OpenEnv-compliant reinforcement learning environment that "
        "simulates real-world negotiation tasks. Supports three difficulty "
        "levels: single-issue price negotiation (easy), multi-issue job "
        "offer negotiation (medium), and multi-vendor contract negotiation (hard)."
    ),
    version="1.0.0",
)

# Allow CORS for HF Spaces and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (stateful, one episode at a time)
env = NegotiationEnvironment()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["health"])
async def root():
    """Health / ping endpoint.  Returns 200 with environment metadata.
    
    This is the endpoint the automated validator pings to confirm the
    Space is alive.
    """
    return HealthResponse()


@app.get("/health", tags=["health"])
async def health():
    """Lightweight health check (used by Docker HEALTHCHECK)."""
    return {"status": "healthy"}


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
async def reset(request: ResetRequest):
    """Reset the environment and start a new episode.

    Args:
        request: ResetRequest with task_id (default "task_1") and
                 optional scenario_id.

    Returns:
        ResetResponse containing the initial NegotiationObservation.
    """
    try:
        observation = env.reset(
            task_id=request.task_id,
            scenario_id=request.scenario_id,
        )
        logger.info(
            "RESET  task=%s  scenario=%s  max_steps=%d",
            request.task_id,
            observation.counterparty_offer.model_dump(exclude_none=True) if observation.counterparty_offer else "N/A",
            observation.max_steps,
        )
        return ResetResponse(observation=observation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during reset")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(request: StepRequest):
    """Execute one negotiation step.

    Args:
        request: StepRequest containing a NegotiationAction.

    Returns:
        StepResponse with observation, reward, done flag, and info dict.
    """
    try:
        observation, reward, done, info = env.step(request.action)
        logger.info(
            "STEP   action=%s  reward=%.4f  done=%s  status=%s",
            request.action.action_type.value,
            reward,
            done,
            observation.status.value,
        )
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except Exception as e:
        logger.exception("Error during step")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/state", response_model=StateResponse, tags=["environment"])
async def state():
    """Return full internal environment state.

    Includes hidden information (opponent constraints, vendor floors).
    Used for grading, debugging, and the OpenEnv state() contract.
    """
    try:
        full_state = env.state()
        return StateResponse(
            state=full_state,
        )
    except Exception as e:
        logger.exception("Error getting state")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/schema", tags=["meta"])
async def schema():
    """Return JSON schemas for Action, Observation, and State models."""
    from models import EnvironmentState
    return {
        "action": NegotiationAction.model_json_schema(),
        "observation": NegotiationObservation.model_json_schema(),
        "state": EnvironmentState.model_json_schema(),
    }


@app.get("/tasks", tags=["meta"])
async def list_tasks():
    """List all available tasks with descriptions and difficulty levels."""
    return {
        "tasks": [
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "name": "Single-Issue Price Negotiation",
                "description": "Buy a used item at the best possible price. Single issue (price), 5 turns.",
                "max_steps": 5,
                "issues": ["price"],
            },
            {
                "task_id": "task_2",
                "difficulty": "medium",
                "name": "Multi-Issue Job Offer Negotiation",
                "description": "Negotiate salary, remote days, and start date for a job offer. 8 turns.",
                "max_steps": 8,
                "issues": ["salary", "remote_days", "start_date_weeks"],
            },
            {
                "task_id": "task_3",
                "difficulty": "hard",
                "name": "Multi-Vendor Contract Negotiation",
                "description": "Negotiate with 2 competing vendors on price, delivery, and payment terms. 12 turns.",
                "max_steps": 12,
                "issues": ["price", "delivery_days", "payment_terms_days"],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
