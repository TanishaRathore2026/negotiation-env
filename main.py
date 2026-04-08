"""
main.py — FastAPI application exposing all OpenEnv endpoints for the
AI Negotiation Environment.

Endpoints:
  GET  /health  → health check
  GET  /tasks   → list available tasks
  POST /reset   → reset environment to a new episode
  POST /step    → submit agent action, receive observation + reward
  GET  /state   → full internal state (hidden info included)
  POST /grade   → run deterministic grader on current state

Run:
  uvicorn main:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from environment import NegotiationEnvironment
from models import (
    NegotiationAction,
    NegotiationObservation,
    NegotiationStatus,
)
from tasks import get_all_tasks, get_task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("negotiation_env")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs on startup and shutdown."""
    # ── Startup ──
    logger.info("=" * 60)
    logger.info("Negotiation Environment ready")
    logger.info("Tasks: %s", [t.task_id for t in get_all_tasks()])
    logger.info("Port:  %s", os.environ.get("PORT", "7860"))
    logger.info("=" * 60)
    yield
    # ── Shutdown ──
    logger.info("Negotiation Environment shutting down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Negotiation Environment",
    description=(
        "An OpenEnv-compliant RL environment simulating real-world "
        "negotiation tasks across 3 difficulty levels: price haggling "
        "(easy), job offer (medium), and vendor contract (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all for HF Spaces and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single stateful environment instance (one episode at a time)
env = NegotiationEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_id: str = Field(
        "task_1",
        description="Which task to load: task_1 (easy), task_2 (medium), task_3 (hard)",
    )
    scenario_id: Optional[str] = Field(
        None,
        description="Optional specific scenario ID. Random if omitted.",
    )


class StepRequest(BaseModel):
    """Body for POST /step."""
    action: NegotiationAction = Field(
        ..., description="The agent's negotiation action",
    )


class StepResponse(BaseModel):
    """Response from POST /step."""
    observation: NegotiationObservation
    reward: float
    done: bool
    info: dict[str, Any]


class GradeRequest(BaseModel):
    """Body for POST /grade."""
    task_id: str = Field(
        "task_1", description="Task to grade: task_1, task_2, or task_3",
    )


class GradeResponse(BaseModel):
    """Response from POST /grade."""
    task_id: str
    score: float


class TaskInfo(BaseModel):
    """Single task in the GET /tasks response."""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_turns: int


# =========================================================================
# ENDPOINTS
# =========================================================================


# ── GET /health ──────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    """Lightweight health / ping endpoint."""
    return {
        "status": "ok",
        "environment": "negotiation-env",
        "version": "1.0.0",
        "tasks": ["task_1", "task_2", "task_3"],
    }


# ── GET / (alias for /health) ───────────────────────────────
@app.get("/", tags=["health"])
async def root():
    """Root endpoint — same as /health for OpenEnv compatibility."""
    return {
        "status": "ok",
        "environment": "negotiation-env",
        "version": "1.0.0",
        "tasks": ["task_1", "task_2", "task_3"],
    }


# ── GET /tasks ───────────────────────────────────────────────
@app.get("/tasks", response_model=list[TaskInfo], tags=["meta"])
async def list_tasks():
    """List all available tasks with descriptions and difficulty levels."""
    tasks = get_all_tasks()
    return [
        TaskInfo(
            task_id=t.task_id,
            name=t.name,
            description=t.description,
            difficulty=t.difficulty,
            max_turns=t.max_turns,
        )
        for t in tasks
    ]


# ── POST /reset ──────────────────────────────────────────────
@app.post("/reset", tags=["environment"])
async def reset(request: Optional[ResetRequest] = None):
    """Reset the environment and begin a new episode."""
    if request is None:
        request = ResetRequest()
    try:
        observation = env.reset(
            task_id=request.task_id,
            scenario_id=request.scenario_id,
        )
        logger.info(
            "RESET  task=%s  max_steps=%d",
            request.task_id,
            observation.max_steps,
        )
        return JSONResponse(
            status_code=200,
            content={"observation": observation.model_dump()},
        )

    except ValueError as exc:
        logger.warning("RESET failed (400): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("RESET failed (500)")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── POST /step ───────────────────────────────────────────────
@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(request: StepRequest):
    """Execute one negotiation step."""
    try:
        observation, reward, done, info = env.step(request.action)

        logger.info(
            "STEP   turn=%d  action=%s  reward=%.4f  done=%s  status=%s",
            observation.current_step,
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

    except Exception as exc:
        logger.exception("STEP failed (500)")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── GET /state ───────────────────────────────────────────────
@app.get("/state", tags=["environment"])
async def state():
    """Return the full internal environment state."""
    try:
        full_state = env.state()
        return JSONResponse(
            status_code=200,
            content={"state": _serialize(full_state)},
        )

    except Exception as exc:
        logger.exception("STATE failed (500)")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── POST /grade ──────────────────────────────────────────────
@app.post("/grade", response_model=GradeResponse, tags=["grading"])
async def grade(request: Optional[GradeRequest] = None):
    """Run the deterministic grader on the current episode."""
    if request is None:
        request = GradeRequest()
    try:
        task = get_task(request.task_id)
        current_state = env.state()
        score = task.grade(current_state)

        logger.info(
            "GRADE  task=%s  score=%.4f  status=%s",
            request.task_id,
            score,
            current_state.get("status", "unknown"),
        )

        return GradeResponse(task_id=request.task_id, score=round(score, 4))

    except ValueError as exc:
        logger.warning("GRADE failed (400): %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("GRADE failed (500)")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# =========================================================================
# Helper: serialise state dicts containing enums
# =========================================================================

def _serialize(obj: Any) -> Any:
    """Recursively convert enums and non-JSON-friendly types to strings."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif hasattr(obj, "value"):
        return obj.value
    return obj


# =========================================================================
# Direct execution
# =========================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)