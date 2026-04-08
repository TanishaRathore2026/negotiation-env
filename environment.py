"""
environment.py — Core OpenEnv environment for AI Negotiation.

Implements:
  • reset(task_id)  → NegotiationObservation   (start a new episode)
  • step(action)    → (observation, reward, done, info)
  • state()         → dict                      (full internal state)

Supports three task difficulties via the scenarios in data/scenarios.json:
  task_1  (easy)   – single-issue price negotiation
  task_2  (medium) – multi-issue job offer negotiation
  task_3  (hard)   – multi-vendor, multi-issue contract negotiation
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional

from models import (
    ActionType,
    CounterpartyOffer,
    EnvironmentState,
    NegotiationAction,
    NegotiationObservation,
    NegotiationStatus,
    RewardBreakdown,
    StepResponse,
    TaskDifficulty,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Path to the bundled scenario bank
_SCENARIO_FILE = Path(__file__).parent / "data" / "scenarios.json"

# Max steps per task difficulty
_MAX_STEPS: dict[str, int] = {
    "task_1": 5,
    "task_2": 8,
    "task_3": 12,
}

# Map task_id → JSON key in scenarios.json
_TASK_KEY: dict[str, str] = {
    "task_1": "task_1_easy",
    "task_2": "task_2_medium",
    "task_3": "task_3_hard",
}

# Map task_id → TaskDifficulty enum
_DIFFICULTY: dict[str, TaskDifficulty] = {
    "task_1": TaskDifficulty.EASY,
    "task_2": TaskDifficulty.MEDIUM,
    "task_3": TaskDifficulty.HARD,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------


class NegotiationEnvironment:
    """OpenEnv-compliant negotiation environment.

    Usage::

        env = NegotiationEnvironment()
        obs = env.reset("task_1")          # start easy scenario
        obs, reward, done, info = env.step(action)
        full_state = env.state()
    """

    def __init__(self) -> None:
        """Initialise the environment (load scenario bank)."""
        # Load all scenarios from JSON once at startup
        with open(_SCENARIO_FILE, "r", encoding="utf-8") as f:
            self._scenario_bank: dict[str, list[dict]] = json.load(f)

        # Internal state — populated by reset()
        self._task_id: str = ""
        self._difficulty: TaskDifficulty = TaskDifficulty.EASY
        self._scenario: dict[str, Any] = {}
        self._max_steps: int = 5
        self._current_step: int = 0
        self._status: NegotiationStatus = NegotiationStatus.IN_PROGRESS
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._last_reward: Optional[RewardBreakdown] = None

        # Track the current offer on the table (from counterparty perspective)
        # and the agent's last offer so we can compute progress
        self._counterparty_offer: Optional[CounterpartyOffer] = None
        self._agent_last_offer: dict[str, Any] = {}

        # For task 3: track per-vendor state
        self._vendor_offers: dict[str, dict[str, Any]] = {}
        self._chosen_vendor: Optional[str] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_1", scenario_id: Optional[str] = None) -> NegotiationObservation:
        """Reset the environment and start a new episode.

        Args:
            task_id: One of "task_1", "task_2", "task_3".
            scenario_id: Optional specific scenario ID.  If omitted a
                         random scenario is chosen from the task's pool.

        Returns:
            NegotiationObservation with the initial state visible to the
            agent (scenario context, counterparty's opening offer, and
            the agent's constraints).
        """
        if task_id not in _TASK_KEY:
            raise ValueError(f"Invalid task_id '{task_id}'. Must be one of {list(_TASK_KEY)}")

        # ---------- pick scenario ----------
        key = _TASK_KEY[task_id]
        pool = self._scenario_bank[key]

        if scenario_id:
            matches = [s for s in pool if s.get("scenario_id") == scenario_id]
            if not matches:
                raise ValueError(f"scenario_id '{scenario_id}' not found in {key}")
            scenario = matches[0]
        else:
            scenario = random.choice(pool)

        # ---------- reset all internal state ----------
        self._task_id = task_id
        self._difficulty = _DIFFICULTY[task_id]
        self._scenario = copy.deepcopy(scenario)
        self._max_steps = _MAX_STEPS[task_id]
        self._current_step = 0
        self._status = NegotiationStatus.IN_PROGRESS
        self._history = []
        self._cumulative_reward = 0.0
        self._last_reward = None
        self._agent_last_offer = {}
        self._chosen_vendor = None
        self._vendor_offers = {}

        # ---------- build opening counterparty offer ----------
        if task_id == "task_1":
            # Single-issue: seller's starting price
            self._counterparty_offer = CounterpartyOffer(
                price=scenario["seller_starting_price"],
            )
        elif task_id == "task_2":
            # Multi-issue: employer's initial values for each issue
            issues = {iss["name"]: iss["employer_initial"] for iss in scenario["issues"]}
            self._counterparty_offer = CounterpartyOffer(
                salary=issues.get("salary"),
                remote_days=int(issues.get("remote_days", 0)),
                start_date_weeks=int(issues.get("start_date_weeks", 0)),
            )
        elif task_id == "task_3":
            # Multi-vendor: store each vendor's opening offer separately
            for v in scenario["vendors"]:
                vid = v["vendor_id"]
                self._vendor_offers[vid] = {
                    "price": v["initial_price"],
                    "delivery_days": v["initial_delivery_days"],
                    "payment_terms_days": v["initial_payment_terms_days"],
                }
            # Show the first vendor's offer by default
            first = scenario["vendors"][0]
            self._counterparty_offer = CounterpartyOffer(
                vendor_id=first["vendor_id"],
                price=first["initial_price"],
                delivery_days=first["initial_delivery_days"],
                payment_terms_days=first["initial_payment_terms_days"],
            )

        # Record the opening offer in history
        self._history.append({
            "step": 0,
            "actor": "counterparty",
            "offer": self._counterparty_offer.model_dump(exclude_none=True),
            "message": "Opening offer presented.",
        })

        return self._build_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: NegotiationAction) -> tuple[NegotiationObservation, float, bool, dict[str, Any]]:
        """Execute one negotiation step.

        Args:
            action: The agent's NegotiationAction.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._status != NegotiationStatus.IN_PROGRESS:
            # Episode already ended — return terminal state with zero reward
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already finished. Call reset()."}

        self._current_step += 1
        reward_bd = RewardBreakdown()

        # ---- dispatch by action type ----
        if action.action_type == ActionType.ACCEPT:
            reward_bd = self._handle_accept()

        elif action.action_type == ActionType.REJECT:
            # Agent walks away — episode ends, no agreement
            self._status = NegotiationStatus.WALKED_AWAY
            reward_bd.failure_penalty = -0.3  # heavy penalty for walking away

        elif action.action_type in (ActionType.PROPOSE, ActionType.COUNTER):
            reward_bd = self._handle_propose(action)

        else:
            # Unknown action type — penalise
            reward_bd.failure_penalty = -0.2

        # ---- apply efficiency penalty: −0.02 per turn used ----
        reward_bd.efficiency_penalty = -0.02 * self._current_step

        # ---- compute total ----
        reward_bd.compute_total()
        reward = reward_bd.total

        # ---- check turn limit ----
        done = self._status != NegotiationStatus.IN_PROGRESS
        if not done and self._current_step >= self._max_steps:
            self._status = NegotiationStatus.FAILED_NO_AGREEMENT
            done = True
            # Additional penalty for running out of turns without agreement
            reward_bd.failure_penalty += -0.15
            reward_bd.compute_total()
            reward = reward_bd.total

        # ---- bookkeeping ----
        self._cumulative_reward += reward
        self._last_reward = reward_bd

        obs = self._build_observation()
        info: dict[str, Any] = {
            "reward_breakdown": reward_bd.model_dump(),
            "cumulative_reward": round(self._cumulative_reward, 4),
        }
        if done:
            info["final_status"] = self._status.value
            # Grader score = cumulative reward clamped to [0,1]
            info["grader_score"] = _clamp(self._cumulative_reward)

        return obs, round(reward, 4), done, info

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Return the full internal environment state.

        Includes hidden information (opponent constraints, vendor floors)
        that is *not* visible to the agent.  This is used for debugging,
        grading, and the OpenEnv state() endpoint.

        Returns:
            Dict serialisation of EnvironmentState.
        """
        env_state = EnvironmentState(
            task_id=self._task_id,
            task_difficulty=self._difficulty,
            current_step=self._current_step,
            max_steps=self._max_steps,
            status=self._status,
            scenario=self._scenario,
            negotiation_history=self._history,
            cumulative_reward=round(self._cumulative_reward, 4),
            last_reward=self._last_reward,
        )
        return env_state.model_dump()

    # ==================================================================
    # INTERNAL: Accept handling
    # ==================================================================

    def _handle_accept(self) -> RewardBreakdown:
        """Handle ActionType.ACCEPT — agent agrees to current offer.

        Returns:
            RewardBreakdown with fairness bonus and optional penalties.
        """
        bd = RewardBreakdown()

        if self._task_id == "task_1":
            bd = self._accept_task1(bd)
        elif self._task_id == "task_2":
            bd = self._accept_task2(bd)
        elif self._task_id == "task_3":
            bd = self._accept_task3(bd)

        if self._status == NegotiationStatus.AGREED:
            self._history.append({
                "step": self._current_step,
                "actor": "agent",
                "action": "accept",
                "message": "Agent accepted the current offer.",
            })

        return bd

    def _accept_task1(self, bd: RewardBreakdown) -> RewardBreakdown:
        """Accept logic for Task 1 (single-issue price)."""
        offer_price = self._counterparty_offer.price
        max_budget = self._scenario["buyer_max_budget"]
        ideal = self._scenario["buyer_ideal_price"]
        seller_min = self._scenario["seller_min_price"]

        if offer_price is None:
            bd.failure_penalty = -0.3
            return bd

        if offer_price > max_budget:
            # Accepting an offer above budget — bad move
            bd.failure_penalty = -0.3
            # Don't end the episode; the acceptance is invalid
            return bd

        # Valid acceptance!
        self._status = NegotiationStatus.AGREED

        # Progress: how close to ideal price (lower = better for buyer)
        # Full range is seller_starting → ideal
        starting = self._scenario["seller_starting_price"]
        total_range = starting - ideal  # e.g. 1400 - 950 = 450
        achieved = starting - offer_price  # how far we pulled the price down
        bd.progress_reward = _clamp(achieved / total_range) * 0.5 if total_range > 0 else 0.25

        # Fairness: is the accepted price within the ZOPA?
        # ZOPA = [seller_min, buyer_max]
        zopa_mid = (seller_min + max_budget) / 2.0
        max_dist = (max_budget - seller_min) / 2.0
        dist = abs(offer_price - zopa_mid)
        bd.fairness_bonus = _clamp(1.0 - dist / max_dist) * 0.2 if max_dist > 0 else 0.1

        return bd

    def _accept_task2(self, bd: RewardBreakdown) -> RewardBreakdown:
        """Accept logic for Task 2 (multi-issue job offer)."""
        co = self._counterparty_offer
        issues = self._scenario["issues"]

        # Check each issue against candidate minimums
        valid = True
        for iss in issues:
            name = iss["name"]
            cand_min = iss["candidate_min"]
            if name == "salary" and co.salary is not None:
                if co.salary < cand_min:
                    valid = False
            elif name == "remote_days" and co.remote_days is not None:
                if co.remote_days < cand_min:
                    valid = False
            elif name == "start_date_weeks" and co.start_date_weeks is not None:
                # For start_date, candidate wants MORE weeks (later start)
                if co.start_date_weeks < cand_min:
                    valid = False

        if not valid:
            bd.failure_penalty = -0.3
            return bd

        # Valid acceptance
        self._status = NegotiationStatus.AGREED

        # Weighted progress across issues
        total_progress = 0.0
        total_fairness = 0.0
        for iss in issues:
            name = iss["name"]
            w = iss["weight"]
            cand_ideal = iss["candidate_ideal"]
            cand_min = iss["candidate_min"]
            emp_initial = iss["employer_initial"]

            # Get the accepted value
            val = self._get_issue_value(co, name)
            if val is None:
                continue

            # Progress: how close to candidate_ideal vs starting point
            full_range = abs(cand_ideal - emp_initial)
            achieved = abs(val - emp_initial)
            issue_progress = _clamp(achieved / full_range) if full_range > 0 else 0.5
            total_progress += w * issue_progress

            # Fairness: distance from midpoint of [employer_min, candidate_ideal]
            emp_min = iss["employer_min"]
            mid = (emp_min + cand_ideal) / 2.0
            max_dist = abs(cand_ideal - emp_min) / 2.0
            dist = abs(val - mid)
            issue_fairness = _clamp(1.0 - dist / max_dist) if max_dist > 0 else 0.5
            total_fairness += w * issue_fairness

        bd.progress_reward = total_progress * 0.5
        bd.fairness_bonus = total_fairness * 0.2
        return bd

    def _accept_task3(self, bd: RewardBreakdown) -> RewardBreakdown:
        """Accept logic for Task 3 (multi-vendor contract)."""
        co = self._counterparty_offer
        if co.vendor_id is None:
            bd.failure_penalty = -0.3
            return bd

        # Check against agent's constraints
        budget = self._scenario["budget"]
        ideal_delivery = self._scenario["ideal_delivery_days"]
        ideal_payment = self._scenario["ideal_payment_terms_days"]

        accepted_price = co.price
        accepted_delivery = co.delivery_days
        accepted_payment = co.payment_terms_days

        if accepted_price is None or accepted_price > budget:
            bd.failure_penalty = -0.3
            return bd

        # Valid acceptance
        self._status = NegotiationStatus.AGREED
        self._chosen_vendor = co.vendor_id

        # Find the vendor data for computing fairness
        vendor_data = None
        for v in self._scenario["vendors"]:
            if v["vendor_id"] == co.vendor_id:
                vendor_data = v
                break

        if vendor_data is None:
            bd.failure_penalty = -0.2
            return bd

        # Progress on price: how far from initial price toward min
        init_price = vendor_data["initial_price"]
        min_price = vendor_data["min_price"]
        price_range = init_price - min_price
        price_achieved = init_price - accepted_price
        price_progress = _clamp(price_achieved / price_range) if price_range > 0 else 0.5

        # Progress on delivery: lower is better for buyer
        init_del = vendor_data["initial_delivery_days"]
        min_del = vendor_data["min_delivery_days"]
        del_range = init_del - min_del
        del_achieved = init_del - (accepted_delivery or init_del)
        del_progress = _clamp(del_achieved / del_range) if del_range > 0 else 0.5

        # Progress on payment: higher is better for buyer
        init_pay = vendor_data["initial_payment_terms_days"]
        max_pay = vendor_data["max_payment_terms_days"]
        pay_range = max_pay - init_pay
        pay_achieved = (accepted_payment or init_pay) - init_pay
        pay_progress = _clamp(pay_achieved / pay_range) if pay_range > 0 else 0.5

        # Weighted average (price most important)
        bd.progress_reward = (0.5 * price_progress + 0.3 * del_progress + 0.2 * pay_progress) * 0.5

        # Fairness: how close is the deal to the midpoint of each dimension
        mid_price = (min_price + budget) / 2.0
        mid_del = (min_del + ideal_delivery) / 2.0
        mid_pay = (init_pay + max_pay) / 2.0

        max_p_dist = abs(budget - min_price) / 2.0
        max_d_dist = abs(ideal_delivery - min_del) / 2.0
        max_t_dist = abs(max_pay - init_pay) / 2.0

        p_fair = _clamp(1 - abs(accepted_price - mid_price) / max_p_dist) if max_p_dist > 0 else 0.5
        d_fair = _clamp(1 - abs((accepted_delivery or init_del) - mid_del) / max_d_dist) if max_d_dist > 0 else 0.5
        t_fair = _clamp(1 - abs((accepted_payment or init_pay) - mid_pay) / max_t_dist) if max_t_dist > 0 else 0.5

        bd.fairness_bonus = (0.5 * p_fair + 0.3 * d_fair + 0.2 * t_fair) * 0.2

        return bd

    # ==================================================================
    # INTERNAL: Propose / Counter handling
    # ==================================================================

    def _handle_propose(self, action: NegotiationAction) -> RewardBreakdown:
        """Handle PROPOSE or COUNTER — agent submits a new offer.

        Steps:
          1. Validate the agent's offer against their own constraints.
          2. Record agent offer in history.
          3. Simulate opponent counter-offer.
          4. Compute progress reward.

        Returns:
            RewardBreakdown for this step.
        """
        bd = RewardBreakdown()

        if self._task_id == "task_1":
            bd = self._propose_task1(action, bd)
        elif self._task_id == "task_2":
            bd = self._propose_task2(action, bd)
        elif self._task_id == "task_3":
            bd = self._propose_task3(action, bd)

        return bd

    def _propose_task1(self, action: NegotiationAction, bd: RewardBreakdown) -> RewardBreakdown:
        """Handle proposal for Task 1 (single-issue price)."""
        proposed_price = action.price
        if proposed_price is None:
            bd.failure_penalty = -0.2
            return bd

        max_budget = self._scenario["buyer_max_budget"]
        ideal = self._scenario["buyer_ideal_price"]
        seller_min = self._scenario["seller_min_price"]
        seller_start = self._scenario["seller_starting_price"]

        # Validate: buyer shouldn't offer more than budget
        if proposed_price > max_budget:
            bd.failure_penalty = -0.2

        # Record agent's offer
        self._agent_last_offer = {"price": proposed_price}
        self._history.append({
            "step": self._current_step,
            "actor": "agent",
            "action": "propose",
            "offer": {"price": proposed_price},
            "reasoning": action.reasoning or "",
        })

        # Simulate seller counter-offer
        current_seller_price = self._counterparty_offer.price or seller_start
        counter_price = self._simulate_opponent_value(
            current_value=current_seller_price,
            target_value=proposed_price,
            floor=seller_min,      # seller never goes below their min
            direction="down",       # seller concedes downward
        )

        self._counterparty_offer = CounterpartyOffer(price=counter_price)

        # Record counterparty's response
        self._history.append({
            "step": self._current_step,
            "actor": "counterparty",
            "action": "counter",
            "offer": {"price": counter_price},
            "message": self._generate_counterparty_message("task_1", counter_price),
        })

        # Progress reward: how much did the gap close?
        initial_gap = seller_start - ideal
        current_gap = counter_price - ideal
        bd.progress_reward = _clamp(1.0 - (current_gap / initial_gap)) * 0.5 if initial_gap > 0 else 0.25

        return bd

    def _propose_task2(self, action: NegotiationAction, bd: RewardBreakdown) -> RewardBreakdown:
        """Handle proposal for Task 2 (multi-issue job offer)."""
        issues = self._scenario["issues"]

        # Build agent's offer dict
        agent_offer: dict[str, Any] = {}
        if action.salary is not None:
            agent_offer["salary"] = action.salary
        if action.remote_days is not None:
            agent_offer["remote_days"] = action.remote_days
        if action.start_date_weeks is not None:
            agent_offer["start_date_weeks"] = action.start_date_weeks

        if not agent_offer:
            bd.failure_penalty = -0.2
            return bd

        # Validate against candidate minimums
        for iss in issues:
            name = iss["name"]
            if name in agent_offer:
                cand_min = iss["candidate_min"]
                # For salary/remote_days: agent wants ≥ min
                # For start_date_weeks: agent wants ≥ min (more weeks)
                if agent_offer[name] < cand_min:
                    bd.failure_penalty = -0.2

        self._agent_last_offer = agent_offer
        self._history.append({
            "step": self._current_step,
            "actor": "agent",
            "action": "propose",
            "offer": agent_offer,
            "reasoning": action.reasoning or "",
        })

        # Simulate employer counter for each issue
        counter: dict[str, Any] = {}
        total_progress = 0.0
        for iss in issues:
            name = iss["name"]
            w = iss["weight"]
            emp_initial = iss["employer_initial"]
            emp_min = iss["employer_min"]
            cand_ideal = iss["candidate_ideal"]

            current_val = self._get_issue_value(self._counterparty_offer, name) or emp_initial
            target_val = agent_offer.get(name, cand_ideal)

            # Determine direction: employer concedes toward candidate ideal
            if name == "salary":
                # Employer concedes upward (higher salary)
                new_val = self._simulate_opponent_value(
                    current_value=current_val, target_value=target_val,
                    floor=emp_min, direction="up",
                )
            elif name == "remote_days":
                # Employer concedes upward (more remote days)
                new_val = self._simulate_opponent_value(
                    current_value=current_val, target_value=target_val,
                    floor=emp_min, direction="up",
                )
                new_val = round(new_val)
            elif name == "start_date_weeks":
                # Employer concedes upward (later start = more weeks)
                new_val = self._simulate_opponent_value(
                    current_value=current_val, target_value=target_val,
                    floor=emp_min, direction="up",
                )
                new_val = round(new_val)
            else:
                new_val = current_val

            counter[name] = new_val

            # Progress on this issue
            full_range = abs(cand_ideal - emp_initial)
            current_gap = abs(cand_ideal - new_val)
            issue_progress = _clamp(1.0 - current_gap / full_range) if full_range > 0 else 0.5
            total_progress += w * issue_progress

        # Update counterparty offer
        self._counterparty_offer = CounterpartyOffer(
            salary=counter.get("salary"),
            remote_days=int(counter.get("remote_days", 0)),
            start_date_weeks=int(counter.get("start_date_weeks", 0)),
        )

        self._history.append({
            "step": self._current_step,
            "actor": "counterparty",
            "action": "counter",
            "offer": counter,
            "message": self._generate_counterparty_message("task_2", counter),
        })

        bd.progress_reward = total_progress * 0.5
        return bd

    def _propose_task3(self, action: NegotiationAction, bd: RewardBreakdown) -> RewardBreakdown:
        """Handle proposal for Task 3 (multi-vendor contract)."""
        target_vendor = action.vendor_id
        if not target_vendor or target_vendor not in self._vendor_offers:
            # If no vendor specified, default to first vendor
            if self._vendor_offers:
                target_vendor = list(self._vendor_offers.keys())[0]
            else:
                bd.failure_penalty = -0.2
                return bd

        vendor_data = None
        for v in self._scenario["vendors"]:
            if v["vendor_id"] == target_vendor:
                vendor_data = v
                break

        if vendor_data is None:
            bd.failure_penalty = -0.2
            return bd

        budget = self._scenario["budget"]
        ideal_delivery = self._scenario["ideal_delivery_days"]
        ideal_payment = self._scenario["ideal_payment_terms_days"]

        # Build agent offer
        agent_offer: dict[str, Any] = {"vendor_id": target_vendor}
        if action.price is not None:
            agent_offer["price"] = action.price
            if action.price > budget:
                bd.failure_penalty = -0.2
        if action.delivery_days is not None:
            agent_offer["delivery_days"] = action.delivery_days
        if action.payment_terms_days is not None:
            agent_offer["payment_terms_days"] = action.payment_terms_days

        self._agent_last_offer = agent_offer
        self._history.append({
            "step": self._current_step,
            "actor": "agent",
            "action": "propose",
            "offer": agent_offer,
            "reasoning": action.reasoning or "",
        })

        # Current vendor offer
        current = self._vendor_offers[target_vendor]

        # Simulate vendor counter-offer for each dimension
        # Price: vendor concedes downward
        new_price = self._simulate_opponent_value(
            current_value=current["price"],
            target_value=agent_offer.get("price", current["price"]),
            floor=vendor_data["min_price"],
            direction="down",
        )
        # Delivery: vendor concedes downward (faster)
        new_delivery = round(self._simulate_opponent_value(
            current_value=current["delivery_days"],
            target_value=agent_offer.get("delivery_days", current["delivery_days"]),
            floor=vendor_data["min_delivery_days"],
            direction="down",
        ))
        # Payment: vendor concedes upward (more generous)
        new_payment = round(self._simulate_opponent_value(
            current_value=current["payment_terms_days"],
            target_value=agent_offer.get("payment_terms_days", current["payment_terms_days"]),
            floor=None,
            direction="up",
            ceiling=vendor_data["max_payment_terms_days"],
        ))

        # Update stored vendor offer
        self._vendor_offers[target_vendor] = {
            "price": new_price,
            "delivery_days": new_delivery,
            "payment_terms_days": new_payment,
        }

        # Set as current counterparty offer
        self._counterparty_offer = CounterpartyOffer(
            vendor_id=target_vendor,
            price=new_price,
            delivery_days=new_delivery,
            payment_terms_days=new_payment,
        )

        counter_dict = {
            "vendor_id": target_vendor,
            "price": new_price,
            "delivery_days": new_delivery,
            "payment_terms_days": new_payment,
        }
        self._history.append({
            "step": self._current_step,
            "actor": "counterparty",
            "action": "counter",
            "offer": counter_dict,
            "message": self._generate_counterparty_message("task_3", counter_dict),
        })

        # Compute progress: weighted across price, delivery, payment
        init_price = vendor_data["initial_price"]
        min_price = vendor_data["min_price"]
        price_range = init_price - min_price
        price_prog = _clamp((init_price - new_price) / price_range) if price_range > 0 else 0.5

        init_del = vendor_data["initial_delivery_days"]
        min_del = vendor_data["min_delivery_days"]
        del_range = init_del - min_del
        del_prog = _clamp((init_del - new_delivery) / del_range) if del_range > 0 else 0.5

        init_pay = vendor_data["initial_payment_terms_days"]
        max_pay = vendor_data["max_payment_terms_days"]
        pay_range = max_pay - init_pay
        pay_prog = _clamp((new_payment - init_pay) / pay_range) if pay_range > 0 else 0.5

        bd.progress_reward = (0.5 * price_prog + 0.3 * del_prog + 0.2 * pay_prog) * 0.5
        return bd

    # ==================================================================
    # INTERNAL: Opponent simulation
    # ==================================================================

    def _simulate_opponent_value(
        self,
        current_value: float,
        target_value: float,
        floor: Optional[float] = None,
        direction: str = "down",
        ceiling: Optional[float] = None,
    ) -> float:
        """Simulate the opponent conceding on a single issue.

        The opponent moves 10–15% of the gap toward the agent's target,
        plus ±2% random noise.  Never violates their own constraint.

        Args:
            current_value: Opponent's current position on this issue.
            target_value: Where the agent wants the issue to go.
            floor: Opponent's absolute minimum (they won't go below this).
            direction: "down" if concession means decreasing value,
                       "up" if concession means increasing value.
            ceiling: Opponent's absolute maximum (they won't go above this).

        Returns:
            Opponent's new value for this issue after the counter-offer.
        """
        gap = target_value - current_value

        # Move 10-15% of the gap toward the agent's target
        concession_rate = random.uniform(0.10, 0.15)

        # Add slight randomness (±2% of the gap)
        noise = random.uniform(-0.02, 0.02) * abs(gap) if gap != 0 else 0

        new_value = current_value + (gap * concession_rate) + noise

        # Enforce constraints based on direction
        if direction == "down" and floor is not None:
            new_value = max(new_value, floor)
        elif direction == "up" and ceiling is not None:
            new_value = min(new_value, ceiling)

        # Also enforce floor in "up" direction (don't go below current)
        if direction == "up" and floor is not None:
            new_value = max(new_value, floor)

        # Round to 2 decimal places for prices, keep integers for days/weeks
        new_value = round(new_value, 2)

        return new_value

    # ==================================================================
    # INTERNAL: Build observation
    # ==================================================================

    def _build_observation(self) -> NegotiationObservation:
        """Construct the NegotiationObservation visible to the agent.

        Returns:
            NegotiationObservation with current scenario context,
            counterparty offer, agent constraints, and history.
        """
        # Build agent constraints dict (what the agent is allowed to know)
        agent_constraints = self._get_agent_constraints()

        return NegotiationObservation(
            task_id=self._task_id,
            task_difficulty=self._difficulty,
            current_step=self._current_step,
            max_steps=self._max_steps,
            status=self._status,
            scenario_description=self._scenario.get("description", ""),
            counterparty_offer=self._counterparty_offer,
            agent_constraints=agent_constraints,
            negotiation_history=self._history,
            counterparty_message=self._history[-1].get("message", "") if self._history else "",
        )

    def _get_agent_constraints(self) -> dict[str, Any]:
        """Extract the agent-visible constraints from the scenario.

        Returns:
            Dict of constraints the agent is allowed to see.
        """
        if self._task_id == "task_1":
            return {
                "role": "buyer",
                "max_budget": self._scenario.get("buyer_max_budget"),
                "ideal_price": self._scenario.get("buyer_ideal_price"),
                "item": self._scenario.get("item"),
            }
        elif self._task_id == "task_2":
            constraints: dict[str, Any] = {
                "role": "candidate",
                "company": self._scenario.get("company"),
                "job_role": self._scenario.get("role"),
            }
            for iss in self._scenario.get("issues", []):
                name = iss["name"]
                constraints[f"{name}_ideal"] = iss["candidate_ideal"]
                constraints[f"{name}_min"] = iss["candidate_min"]
                constraints[f"{name}_weight"] = iss["weight"]
            return constraints
        elif self._task_id == "task_3":
            constraints = {
                "role": "buyer",
                "project": self._scenario.get("project"),
                "budget": self._scenario.get("budget"),
                "ideal_delivery_days": self._scenario.get("ideal_delivery_days"),
                "ideal_payment_terms_days": self._scenario.get("ideal_payment_terms_days"),
                "vendors": [],
            }
            # Show vendor names and their current offers, but NOT their floors
            for v in self._scenario.get("vendors", []):
                vendor_info = {
                    "vendor_id": v["vendor_id"],
                    "vendor_name": v["vendor_name"],
                    "current_offer": self._vendor_offers.get(v["vendor_id"], {}),
                }
                constraints["vendors"].append(vendor_info)
            return constraints
        return {}

    # ==================================================================
    # INTERNAL: Helpers
    # ==================================================================

    def _get_issue_value(self, co: CounterpartyOffer, issue_name: str) -> Optional[float]:
        """Extract a numeric value from CounterpartyOffer by issue name.

        Args:
            co: The CounterpartyOffer to read from.
            issue_name: One of 'salary', 'remote_days', 'start_date_weeks',
                        'price', 'delivery_days', 'payment_terms_days'.

        Returns:
            The numeric value, or None if not set.
        """
        mapping = {
            "salary": co.salary,
            "remote_days": co.remote_days,
            "start_date_weeks": co.start_date_weeks,
            "price": co.price,
            "delivery_days": co.delivery_days,
            "payment_terms_days": co.payment_terms_days,
        }
        val = mapping.get(issue_name)
        return float(val) if val is not None else None

    def _generate_counterparty_message(self, task_id: str, counter_data: Any) -> str:
        """Generate a simple natural-language message from the counterparty.

        Makes the negotiation feel more realistic than raw numbers.

        Args:
            task_id: Current task identifier.
            counter_data: The counter-offer value or dict.

        Returns:
            A human-readable counterparty response string.
        """
        messages_task1 = [
            "I can come down a little, but that's a quality machine.",
            "I've already lowered the price. This is getting close to my bottom line.",
            "I appreciate your offer. Here's what I can do.",
            "Let's meet somewhere reasonable.",
            "I think this is a fair counter. Take it or leave it.",
        ]
        messages_task2 = [
            "We've reviewed your request. Here's our updated package.",
            "We can adjust some terms — see our revised offer.",
            "We want to make this work. Here's what we can offer.",
            "Let's find a middle ground on the outstanding items.",
            "We've discussed internally and can move on a few points.",
        ]
        messages_task3 = [
            "Given the scope of the project, here's our revised proposal.",
            "We're flexible on timeline if the price works for you.",
            "We've trimmed our margin to stay competitive. Updated offer below.",
            "We value long-term partnerships — here's an improved bid.",
            "Considering market rates, this is quite competitive.",
        ]

        if task_id == "task_1":
            return random.choice(messages_task1)
        elif task_id == "task_2":
            return random.choice(messages_task2)
        elif task_id == "task_3":
            return random.choice(messages_task3)
        return "Here is our counter-offer."
