"""
models.py — Pydantic v2 typed models for the AI Negotiation Environment.

Defines:
  - Action models (what the agent sends)
  - Observation models (what the agent sees)
  - Reward model (structured reward breakdown)
  - State model (full internal state for state() endpoint)
  - Scenario models (typed scenario definitions loaded from JSON)
  - Envelope models (API response wrappers)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class TaskDifficulty(str, Enum):
    """Difficulty levels mapping to the 3 required tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    """The type of negotiation move the agent can make."""
    PROPOSE = "propose"        # Make a new offer
    COUNTER = "counter"        # Counter the opponent's offer
    ACCEPT = "accept"          # Accept the current offer on the table
    REJECT = "reject"          # Reject and end negotiation (walk away)


class NegotiationStatus(str, Enum):
    """Current status of the negotiation episode."""
    IN_PROGRESS = "in_progress"
    AGREED = "agreed"              # Both parties reached agreement
    FAILED_NO_AGREEMENT = "failed_no_agreement"  # Ran out of turns
    WALKED_AWAY = "walked_away"    # Agent chose to reject/walk away


# ─────────────────────────────────────────────
# Scenario models (loaded from scenarios.json)
# ─────────────────────────────────────────────

class PriceScenario(BaseModel):
    """Task 1 (Easy): Single-item price negotiation scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    item: str = Field(..., description="Item being negotiated")
    seller_min_price: float = Field(..., description="Seller's minimum acceptable price (hidden from agent)")
    seller_starting_price: float = Field(..., description="Seller's initial asking price (visible)")
    buyer_max_budget: float = Field(..., description="Buyer's maximum budget (known to agent)")
    buyer_ideal_price: float = Field(..., description="Buyer's ideal target price (known to agent)")
    description: str = Field(..., description="Human-readable scenario description")


class JobIssue(BaseModel):
    """A single negotiable issue in a job offer (Task 2)."""
    name: str = Field(..., description="Issue name: salary, remote_days, or start_date_weeks")
    employer_initial: float = Field(..., description="Employer's opening offer for this issue")
    employer_min: float = Field(..., description="Employer's minimum acceptable value (hidden)")
    candidate_ideal: float = Field(..., description="Candidate's ideal value (known to agent)")
    candidate_min: float = Field(..., description="Candidate's minimum acceptable value (known to agent)")
    weight: float = Field(..., description="Importance weight for scoring (0.0–1.0)")


class JobOfferScenario(BaseModel):
    """Task 2 (Medium): Multi-issue job offer negotiation scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    company: str = Field(..., description="Company name")
    role: str = Field(..., description="Job role title")
    issues: list[JobIssue] = Field(..., description="List of negotiable issues")
    description: str = Field(..., description="Human-readable scenario description")


class VendorProfile(BaseModel):
    """A single vendor in a multi-party negotiation (Task 3)."""
    vendor_id: str = Field(..., description="Unique vendor identifier")
    vendor_name: str = Field(..., description="Vendor company name")
    initial_price: float = Field(..., description="Vendor's opening price")
    min_price: float = Field(..., description="Vendor's floor price (hidden)")
    initial_delivery_days: int = Field(..., description="Vendor's initial delivery timeline in days")
    min_delivery_days: int = Field(..., description="Vendor's fastest possible delivery (hidden)")
    initial_payment_terms_days: int = Field(..., description="Vendor's initial payment terms in days")
    max_payment_terms_days: int = Field(..., description="Vendor's most generous payment terms (hidden)")


class VendorContractScenario(BaseModel):
    """Task 3 (Hard): Multi-party vendor contract negotiation scenario."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    project: str = Field(..., description="Project name")
    budget: float = Field(..., description="Total available budget (known to agent)")
    ideal_delivery_days: int = Field(..., description="Ideal delivery timeline (known to agent)")
    ideal_payment_terms_days: int = Field(..., description="Ideal payment terms (known to agent)")
    vendors: list[VendorProfile] = Field(..., description="List of vendors to negotiate with")
    description: str = Field(..., description="Human-readable scenario description")


# ─────────────────────────────────────────────
# Action models (what the agent sends)
# ─────────────────────────────────────────────

class NegotiationAction(BaseModel):
    """
    The agent's action at each step.

    For PROPOSE/COUNTER:
      - For Task 1: set `price` field
      - For Task 2: set `salary`, `remote_days`, `start_date_weeks`
      - For Task 3: set `vendor_id` + `price`, `delivery_days`, `payment_terms_days`
    For ACCEPT/REJECT: only `action_type` is needed.
    """
    action_type: ActionType = Field(..., description="Type of negotiation move")
    # Task 1 fields
    price: Optional[float] = Field(None, description="Proposed price (Task 1 & 3)")
    # Task 2 fields
    salary: Optional[float] = Field(None, description="Proposed salary (Task 2)")
    remote_days: Optional[int] = Field(None, description="Proposed remote days per week (Task 2)")
    start_date_weeks: Optional[int] = Field(None, description="Proposed start date in weeks (Task 2)")
    # Task 3 fields
    vendor_id: Optional[str] = Field(None, description="Which vendor this action targets (Task 3)")
    delivery_days: Optional[int] = Field(None, description="Proposed delivery days (Task 3)")
    payment_terms_days: Optional[int] = Field(None, description="Proposed payment terms in days (Task 3)")
    # Optional reasoning (for LLM transparency)
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for this action")


# ─────────────────────────────────────────────
# Observation models (what the agent sees)
# ─────────────────────────────────────────────

class CounterpartyOffer(BaseModel):
    """The latest offer from the counterparty (opponent)."""
    price: Optional[float] = Field(None, description="Counterparty's current price offer")
    salary: Optional[float] = Field(None, description="Counterparty's salary offer (Task 2)")
    remote_days: Optional[int] = Field(None, description="Counterparty's remote days offer (Task 2)")
    start_date_weeks: Optional[int] = Field(None, description="Counterparty's start date offer (Task 2)")
    delivery_days: Optional[int] = Field(None, description="Counterparty's delivery days (Task 3)")
    payment_terms_days: Optional[int] = Field(None, description="Counterparty's payment terms (Task 3)")
    vendor_id: Optional[str] = Field(None, description="Which vendor made this offer (Task 3)")


class NegotiationObservation(BaseModel):
    """
    What the agent observes after each step (or on reset).

    Contains the current state of the negotiation visible to the agent,
    including the counterparty's latest offer and agent's own constraints.
    """
    task_id: str = Field(..., description="Current task identifier")
    task_difficulty: TaskDifficulty = Field(..., description="easy / medium / hard")
    current_step: int = Field(..., description="Current turn number (0-indexed)")
    max_steps: int = Field(..., description="Maximum allowed turns")
    status: NegotiationStatus = Field(..., description="Current negotiation status")
    scenario_description: str = Field(..., description="Human-readable scenario context")
    # What the counterparty is currently offering
    counterparty_offer: Optional[CounterpartyOffer] = Field(
        None, description="The counterparty's latest offer on the table"
    )
    # Agent's known constraints (what the agent is allowed to know)
    agent_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent's own constraints/preferences (budget, ideal values, etc.)"
    )
    # History of offers for context
    negotiation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological list of all offers made by both sides"
    )
    # Message from counterparty (simulated natural language response)
    counterparty_message: str = Field(
        "", description="Natural language message from the counterparty"
    )


# ─────────────────────────────────────────────
# Reward model (structured, multi-component)
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """
    Multi-component reward returned at every step.
    Total reward is always clamped to [0.0, 1.0].
    """
    progress_reward: float = Field(
        0.0, description="How much closer to agreement this step moved (0.0–0.5)"
    )
    efficiency_penalty: float = Field(
        0.0, description="Small penalty per turn used (-0.02 per turn)"
    )
    fairness_bonus: float = Field(
        0.0, description="Bonus if deal is near ZOPA midpoint (0.0–0.2)"
    )
    failure_penalty: float = Field(
        0.0, description="Penalty for proposals outside constraints (-0.2)"
    )
    total: float = Field(
        0.0, description="Final clamped reward in [0.0, 1.0]"
    )

    def compute_total(self) -> float:
        """Calculate and clamp the total reward."""
        raw = self.progress_reward + self.efficiency_penalty + self.fairness_bonus + self.failure_penalty
        self.total = max(0.0, min(1.0, raw))
        return self.total


# ─────────────────────────────────────────────
# State model (full internal state for state())
# ─────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """
    Full internal state returned by state() endpoint.
    Includes hidden information not shown to the agent.
    """
    task_id: str = Field(..., description="Current task identifier")
    task_difficulty: TaskDifficulty = Field(..., description="Difficulty level")
    current_step: int = Field(..., description="Current turn number")
    max_steps: int = Field(..., description="Maximum turns allowed")
    status: NegotiationStatus = Field(..., description="Negotiation status")
    # Full scenario including hidden values
    scenario: dict[str, Any] = Field(
        default_factory=dict, description="Complete scenario data including hidden values"
    )
    # All offers made
    negotiation_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Full negotiation history"
    )
    # Cumulative reward across the episode
    cumulative_reward: float = Field(0.0, description="Sum of all rewards so far")
    # Last reward breakdown
    last_reward: Optional[RewardBreakdown] = Field(
        None, description="Most recent reward breakdown"
    )


# ─────────────────────────────────────────────
# API envelope models (response wrappers)
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: str = Field("task_1", description="Which task to load: task_1, task_2, task_3")
    scenario_id: Optional[str] = Field(
        None, description="Specific scenario ID (optional, random if omitted)"
    )


class StepRequest(BaseModel):
    """Request body for POST /step."""
    action: NegotiationAction = Field(..., description="The agent's negotiation action")


class StepResponse(BaseModel):
    """Response body from POST /step (OpenEnv spec)."""
    observation: NegotiationObservation = Field(..., description="Updated observation")
    reward: float = Field(..., description="Reward for this step (0.0–1.0)")
    done: bool = Field(..., description="Whether the episode is finished")
    info: dict[str, Any] = Field(
        default_factory=dict, description="Additional info (reward breakdown, grader score, etc.)"
    )


class ResetResponse(BaseModel):
    """Response body from POST /reset."""
    observation: NegotiationObservation = Field(..., description="Initial observation")


class StateResponse(BaseModel):
    """Response body from GET /state."""
    state: EnvironmentState = Field(..., description="Full internal environment state")


class HealthResponse(BaseModel):
    """Response body from GET / (ping/health)."""
    status: str = Field("OpenEnv Live", description="Service status")
    environment: str = Field("AI Negotiation Environment", description="Environment name")
    version: str = Field("1.0.0", description="Environment version")
    tasks: list[str] = Field(
        default=["task_1", "task_2", "task_3"],
        description="Available task IDs"
    )
