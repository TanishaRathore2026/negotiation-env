"""
tasks.py — Task definitions and deterministic graders for the AI Negotiation Environment.

Defines 3 tasks with increasing difficulty, each containing a
programmatic grader that produces a score in [0.0, 1.0]:

  Task 1 (easy)   — Single-issue price negotiation
  Task 2 (medium) — Multi-issue job offer negotiation
  Task 3 (hard)   — Multi-vendor contract negotiation

Graders are 100% DETERMINISTIC — identical inputs always produce
identical outputs.  No randomness anywhere in this file.

Usage:
    from tasks import get_task, get_all_tasks

    task = get_task("task_1")
    score = task.grade(env.state())   # → float in [0.0, 1.0]
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helper: clamp to [0.0, 1.0]
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi].  Deterministic, no side effects."""
    return max(lo, min(hi, value))


def _normalize_status(status: Any) -> str:
    """Normalize the status field to a plain lowercase string.

    Handles:
      - NegotiationStatus enum objects (have .value)
      - Strings like "NegotiationStatus.AGREED"
      - Plain strings like "agreed"

    Returns:
        Lowercase status string: "agreed", "in_progress",
        "failed_no_agreement", "walked_away".
    """
    if hasattr(status, "value"):
        # It's an enum — extract the .value string
        return str(status.value).lower()
    s = str(status).lower()
    # Strip enum class prefix if present, e.g. "negotiationstatus.agreed" → "agreed"
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s


# ---------------------------------------------------------------------------
# Helper: extract last counterparty offer from negotiation history
# ---------------------------------------------------------------------------

def _last_counterparty_offer(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the most recent counterparty offer from the history.

    Walks history in reverse and returns the first entry where
    actor == 'counterparty' and an 'offer' key exists.
    Returns empty dict if none found.
    """
    for entry in reversed(history):
        if entry.get("actor") == "counterparty" and "offer" in entry:
            return entry["offer"]
    return {}


def _last_agent_offer(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the most recent agent offer from the history."""
    for entry in reversed(history):
        if entry.get("actor") == "agent" and "offer" in entry:
            return entry["offer"]
    return {}


def _collect_vendor_last_offers(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collect the latest counterparty offer for each vendor_id."""
    vendor_offers: dict[str, dict[str, Any]] = {}
    for entry in history:
        if entry.get("actor") == "counterparty" and "offer" in entry:
            offer = entry["offer"]
            vid = offer.get("vendor_id")
            if vid:
                vendor_offers[vid] = offer
    return vendor_offers


# =========================================================================
# Base Task class
# =========================================================================

class Task:
    """Base class for all negotiation tasks.

    Attributes:
        task_id:     Unique identifier ("task_1", "task_2", "task_3").
        name:        Human-readable task name.
        description: What the agent must accomplish.
        difficulty:  "easy", "medium", or "hard".
        max_turns:   Maximum negotiation turns allowed.
        scenario_id: Default scenario to use (can be overridden).
    """

    task_id: str = ""
    name: str = ""
    description: str = ""
    difficulty: str = ""
    max_turns: int = 5
    scenario_id: str = ""

    def grade(self, state: dict[str, Any]) -> float:
        """Grade a completed (or timed-out) episode.

        Args:
            state: The full state dict from env.state().
                   Contains: task_id, task_difficulty, current_step,
                   max_steps, status, scenario, negotiation_history,
                   cumulative_reward, last_reward.

        Returns:
            Deterministic score in [0.0, 1.0].
        """
        raise NotImplementedError("Subclasses must implement grade()")


# =========================================================================
# Task 1 — Simple Price Negotiation (Easy)
# =========================================================================

class Task1_SimplePriceNegotiation(Task):
    """Task 1: Single-issue price negotiation.

    The agent plays a buyer negotiating the purchase price of an item.
    Success = buy at or below budget, ideally near the ZOPA midpoint.

    Scoring breakdown:
      • If NOT agreed → partial credit (max 0.3) based on price gap closed
      • If agreed:
        - proximity score: how close to the ZOPA midpoint (0.0–1.0)
        - efficiency bonus: fewer turns → higher score
        - penalty if accepted above budget
    """

    task_id = "task_1"
    name = "Single-Issue Price Negotiation"
    description = (
        "Buy a used item at the best possible price. "
        "Negotiate a single issue (price) within 5 turns."
    )
    difficulty = "easy"
    max_turns = 5
    scenario_id = "price_001"

    def grade(self, state: dict[str, Any]) -> float:
        """Grade a Task 1 episode.  100% deterministic."""
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)

        # Extract scenario parameters
        seller_start = scenario.get("seller_starting_price", 0)
        seller_min = scenario.get("seller_min_price", 0)
        buyer_max = scenario.get("buyer_max_budget", 0)
        buyer_ideal = scenario.get("buyer_ideal_price", 0)

        # Compute ZOPA midpoint: fair deal = midpoint of [seller_min, buyer_max]
        zopa_mid = (seller_min + buyer_max) / 2.0

        # ── NOT AGREED: partial credit only ──
        if status != "agreed":
            # How much did the price gap close?
            last_cp = _last_counterparty_offer(history)
            final_cp_price = last_cp.get("price", seller_start)

            # Initial gap: distance from seller start to buyer ideal
            initial_gap = seller_start - buyer_ideal
            # Current gap: distance from last seller offer to buyer ideal
            current_gap = final_cp_price - buyer_ideal

            if initial_gap <= 0:
                # Edge case: seller started at or below ideal (shouldn't happen)
                progress = 0.5
            else:
                # How much of the initial gap was closed?
                # 1.0 = fully closed (seller at ideal), 0.0 = no progress
                progress = _clamp(1.0 - (current_gap / initial_gap))

            # Partial credit capped at 0.3 — you can't score well without a deal
            return round(_clamp(progress * 0.3), 4)

        # ── AGREED: score the deal ──
        last_cp = _last_counterparty_offer(history)
        final_price = last_cp.get("price", seller_start)

        # Penalty: if agent somehow accepted above budget
        if final_price > buyer_max:
            # Bad deal — accepted above budget constraint
            return 0.1

        # --- Proximity score: how close to ZOPA midpoint? ---
        # Best case: final_price == zopa_mid → proximity = 1.0
        # Worst case: final_price == buyer_max → proximity ≈ 0.0
        # We measure distance from zopa midpoint, normalised by the
        # half-width of the ZOPA range
        zopa_half_width = (buyer_max - seller_min) / 2.0
        if zopa_half_width > 0:
            distance = abs(final_price - zopa_mid)
            proximity = _clamp(1.0 - (distance / zopa_half_width))
        else:
            proximity = 0.5

        # --- Efficiency bonus: fewer turns = better ---
        # Lose 0.04 per turn used (max loss = 0.2 at 5 turns)
        if max_steps > 0:
            efficiency = _clamp(1.0 - (current_step / max_steps) * 0.2)
        else:
            efficiency = 1.0

        # --- Deal quality bonus: how much below budget? ---
        # Getting a price well below budget is good
        budget_range = buyer_max - buyer_ideal
        if budget_range > 0:
            savings = _clamp((buyer_max - final_price) / budget_range)
        else:
            savings = 0.5

        # Composite score:
        #   proximity_weight = 0.50  (how fair/good is the deal)
        #   savings_weight   = 0.25  (how much did agent save)
        #   efficiency_weight = 0.25 (how fast)
        composite = (0.50 * proximity) + (0.25 * savings) + (0.25 * efficiency)

        return round(_clamp(composite), 4)


# =========================================================================
# Task 2 — Job Offer Negotiation (Medium)
# =========================================================================

class Task2_JobOfferNegotiation(Task):
    """Task 2: Multi-issue job offer negotiation.

    The agent plays a candidate negotiating salary, remote days,
    and start date.  Issues have different weights reflecting their
    importance.

    Scoring breakdown:
      • Per-issue score: how close to candidate_ideal vs candidate_min
      • Weighted by issue importance (salary=0.5, remote=0.3, start=0.2)
      • Efficiency penalty: −0.02 per turn used
      • If NOT agreed: partial credit capped at 0.35
    """

    task_id = "task_2"
    name = "Multi-Issue Job Offer Negotiation"
    description = (
        "Negotiate a job offer: salary, remote days, and start date. "
        "Balance trade-offs across 3 issues within 8 turns."
    )
    difficulty = "medium"
    max_turns = 8
    scenario_id = "job_001"

    def grade(self, state: dict[str, Any]) -> float:
        """Grade a Task 2 episode.  100% deterministic."""
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)
        issues = scenario.get("issues", [])

        if not issues:
            return 0.0

        # ── NOT AGREED: partial credit based on how far the employer moved ──
        if status != "agreed":
            total_progress = 0.0
            total_weight = 0.0

            last_cp = _last_counterparty_offer(history)

            for iss in issues:
                name = iss["name"]
                weight = iss.get("weight", 0.33)
                emp_initial = iss["employer_initial"]
                cand_ideal = iss["candidate_ideal"]

                # Get the last counterparty value for this issue
                current_val = last_cp.get(name, emp_initial)

                # How much of the gap was closed?
                full_range = abs(cand_ideal - emp_initial)
                if full_range > 0:
                    moved = abs(current_val - emp_initial)
                    issue_progress = _clamp(moved / full_range)
                else:
                    issue_progress = 0.5

                total_progress += weight * issue_progress
                total_weight += weight

            # Normalise by total weight
            if total_weight > 0:
                weighted_progress = total_progress / total_weight
            else:
                weighted_progress = 0.0

            # Partial credit capped at 0.35
            return round(_clamp(weighted_progress * 0.35), 4)

        # ── AGREED: score each issue ──
        last_cp = _last_counterparty_offer(history)

        total_score = 0.0
        total_weight = 0.0

        for iss in issues:
            name = iss["name"]
            weight = iss.get("weight", 0.33)
            cand_ideal = iss["candidate_ideal"]
            cand_min = iss["candidate_min"]
            emp_initial = iss["employer_initial"]

            # Get the accepted value
            accepted_val = last_cp.get(name)
            if accepted_val is None:
                accepted_val = emp_initial

            # --- Issue score: how favorable is the deal for the candidate? ---
            # Best case: accepted == candidate_ideal → 1.0
            # Okay case: accepted == candidate_min → 0.3
            # Bad case:  accepted == employer_initial → 0.0
            ideal_range = abs(cand_ideal - cand_min)
            full_range = abs(cand_ideal - emp_initial)

            if full_range > 0:
                # How far from employer_initial toward candidate_ideal?
                achieved = abs(accepted_val - emp_initial)
                issue_quality = _clamp(achieved / full_range)
            else:
                issue_quality = 0.5

            # --- Fairness sub-score: how close to midpoint of ZOPA? ---
            emp_min = iss.get("employer_min", emp_initial)
            zopa_mid = (emp_min + cand_ideal) / 2.0
            zopa_half = abs(cand_ideal - emp_min) / 2.0
            if zopa_half > 0:
                fairness = _clamp(1.0 - abs(accepted_val - zopa_mid) / zopa_half)
            else:
                fairness = 0.5

            # Combined issue score: 70% quality + 30% fairness
            issue_score = 0.70 * issue_quality + 0.30 * fairness

            total_score += weight * issue_score
            total_weight += weight

        # Normalise
        if total_weight > 0:
            weighted_score = total_score / total_weight
        else:
            weighted_score = 0.0

        # --- Efficiency penalty: −0.02 per turn used ---
        # Using all 8 turns = −0.16 penalty
        efficiency_penalty = 0.02 * current_step

        # Final score
        final = weighted_score - efficiency_penalty

        return round(_clamp(final), 4)


# =========================================================================
# Task 3 — Vendor Contract Negotiation (Hard)
# =========================================================================

class Task3_VendorContractNegotiation(Task):
    """Task 3: Multi-vendor, multi-issue contract negotiation.

    The agent negotiates with 2 competing vendors on price, delivery
    days, and payment terms, then selects the best deal.

    Scoring breakdown:
      • deal_quality    (weight 0.4): How good is the accepted deal?
      • efficiency      (weight 0.3): How quickly was agreement reached?
      • vendor_selection (weight 0.3): Did agent pick the better vendor?
      • If NOT agreed: partial credit capped at 0.25
    """

    task_id = "task_3"
    name = "Multi-Vendor Contract Negotiation"
    description = (
        "Negotiate with 2 competing vendors on price, delivery "
        "timeline, and payment terms. Pick the best overall deal. "
        "12 turns maximum."
    )
    difficulty = "hard"
    max_turns = 12
    scenario_id = "vendor_001"

    def grade(self, state: dict[str, Any]) -> float:
        """Grade a Task 3 episode.  100% deterministic."""
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)
        vendors = scenario.get("vendors", [])

        if not vendors:
            return 0.0

        budget = scenario.get("budget", 0)
        ideal_delivery = scenario.get("ideal_delivery_days", 0)
        ideal_payment = scenario.get("ideal_payment_terms_days", 0)

        # ── NOT AGREED: partial credit based on progress across vendors ──
        if status != "agreed":
            vendor_last_offers = _collect_vendor_last_offers(history)
            if not vendor_last_offers:
                return 0.0

            best_progress = 0.0
            for vid, offer in vendor_last_offers.items():
                v_data = self._find_vendor(vendors, vid)
                if not v_data:
                    continue
                progress = self._vendor_progress(v_data, offer)
                best_progress = max(best_progress, progress)

            # Partial credit capped at 0.25 — hard task demands agreement
            return round(_clamp(best_progress * 0.25), 4)

        # ── AGREED: full scoring ──

        # Determine which vendor was chosen (from last accepted offer)
        last_cp = _last_counterparty_offer(history)
        chosen_vid = last_cp.get("vendor_id")
        chosen_vendor = self._find_vendor(vendors, chosen_vid)

        if not chosen_vendor:
            return 0.1  # Agreed but can't identify vendor — minimal credit

        # ═══════════════════════════════════════════════════════════
        # Component 1: DEAL QUALITY (weight = 0.4)
        # How good is the accepted deal on each dimension?
        # ═══════════════════════════════════════════════════════════

        accepted_price = last_cp.get("price", chosen_vendor["initial_price"])
        accepted_delivery = last_cp.get("delivery_days", chosen_vendor["initial_delivery_days"])
        accepted_payment = last_cp.get("payment_terms_days", chosen_vendor["initial_payment_terms_days"])

        # Price quality: lower = better for buyer
        # Score 1.0 if price == vendor's min_price, 0.0 if price == budget
        min_p = chosen_vendor["min_price"]
        if budget - min_p > 0:
            price_quality = _clamp(1.0 - (accepted_price - min_p) / (budget - min_p))
        else:
            price_quality = 0.5

        # Delivery quality: closer to ideal = better (lower delivery = better)
        init_d = chosen_vendor["initial_delivery_days"]
        min_d = chosen_vendor["min_delivery_days"]
        if init_d - min_d > 0:
            delivery_quality = _clamp((init_d - accepted_delivery) / (init_d - min_d))
        else:
            delivery_quality = 0.5

        # Payment quality: higher = better for buyer (more time to pay)
        init_t = chosen_vendor["initial_payment_terms_days"]
        max_t = chosen_vendor["max_payment_terms_days"]
        if max_t - init_t > 0:
            payment_quality = _clamp((accepted_payment - init_t) / (max_t - init_t))
        else:
            payment_quality = 0.5

        # Weighted deal quality (price is most important in procurement)
        deal_quality = (
            0.50 * price_quality +
            0.30 * delivery_quality +
            0.20 * payment_quality
        )

        # ═══════════════════════════════════════════════════════════
        # Component 2: NEGOTIATION EFFICIENCY (weight = 0.3)
        # Fewer turns = higher score
        # ═══════════════════════════════════════════════════════════

        if max_steps > 1:
            # Score 1.0 at turn 1, linearly down to 0.2 at max turns
            efficiency = _clamp(1.0 - 0.8 * ((current_step - 1) / (max_steps - 1)))
        else:
            efficiency = 1.0

        # ═══════════════════════════════════════════════════════════
        # Component 3: VENDOR SELECTION BONUS (weight = 0.3)
        # Did the agent pick the objectively better vendor?
        # ═══════════════════════════════════════════════════════════

        vendor_selection = self._score_vendor_selection(
            vendors=vendors,
            chosen_vid=chosen_vid,
            history=history,
            budget=budget,
            ideal_delivery=ideal_delivery,
            ideal_payment=ideal_payment,
        )

        # ═══════════════════════════════════════════════════════════
        # Composite Score
        # ═══════════════════════════════════════════════════════════

        composite = (
            0.40 * deal_quality +
            0.30 * efficiency +
            0.30 * vendor_selection
        )

        return round(_clamp(composite), 4)

    # ----- internal helpers (all deterministic) -----

    @staticmethod
    def _find_vendor(
        vendors: list[dict[str, Any]], vendor_id: str | None
    ) -> dict[str, Any] | None:
        """Find vendor data by vendor_id.  Deterministic lookup."""
        if not vendor_id:
            return None
        for v in vendors:
            if v.get("vendor_id") == vendor_id:
                return v
        return None

    @staticmethod
    def _vendor_progress(v_data: dict[str, Any], offer: dict[str, Any]) -> float:
        """Compute how much progress was made negotiating with one vendor.

        Returns float in [0.0, 1.0].  Deterministic.
        """
        # Price progress (lower = better for buyer)
        init_p = v_data["initial_price"]
        min_p = v_data["min_price"]
        cur_p = offer.get("price", init_p)
        p_range = init_p - min_p
        p_prog = _clamp((init_p - cur_p) / p_range) if p_range > 0 else 0.5

        # Delivery progress (lower = better)
        init_d = v_data["initial_delivery_days"]
        min_d = v_data["min_delivery_days"]
        cur_d = offer.get("delivery_days", init_d)
        d_range = init_d - min_d
        d_prog = _clamp((init_d - cur_d) / d_range) if d_range > 0 else 0.5

        # Payment progress (higher = better for buyer)
        init_t = v_data["initial_payment_terms_days"]
        max_t = v_data["max_payment_terms_days"]
        cur_t = offer.get("payment_terms_days", init_t)
        t_range = max_t - init_t
        t_prog = _clamp((cur_t - init_t) / t_range) if t_range > 0 else 0.5

        # Weighted average (price most important in procurement)
        return 0.50 * p_prog + 0.30 * d_prog + 0.20 * t_prog

    def _score_vendor_selection(
        self,
        vendors: list[dict[str, Any]],
        chosen_vid: str | None,
        history: list[dict[str, Any]],
        budget: float,
        ideal_delivery: float,
        ideal_payment: float,
    ) -> float:
        """Score whether the agent picked the objectively better vendor.

        We compute a "best possible deal" score for each vendor based on
        their constraint floors (the best deal each vendor COULD offer),
        then check if the agent chose the one with the higher potential.

        If only one vendor was negotiated with, give 0.5 (neutral).
        If the agent picked the better vendor, give 1.0.
        If the agent picked the worse vendor, give 0.3 (still some credit).

        Returns float in [0.0, 1.0].  Deterministic.
        """
        if len(vendors) < 2 or not chosen_vid:
            return 0.5  # Can't evaluate selection with < 2 vendors

        # Compute "best possible deal" score for each vendor
        # This uses the vendor's FLOOR values (hidden from agent)
        vendor_scores: dict[str, float] = {}
        for v in vendors:
            vid = v["vendor_id"]

            # Best price achievable (vendor's floor)
            best_price = v["min_price"]
            best_delivery = v["min_delivery_days"]
            best_payment = v["max_payment_terms_days"]

            # Score each dimension: how good is the vendor's best vs agent's ideal?
            # Price: lower percentage of budget = better
            if budget > 0:
                p_score = _clamp(1.0 - best_price / budget)
            else:
                p_score = 0.5

            # Delivery: closer to ideal = better
            if ideal_delivery > 0:
                d_score = _clamp(1.0 - abs(best_delivery - ideal_delivery) / ideal_delivery)
            else:
                d_score = 0.5

            # Payment: how close to ideal payment terms
            if ideal_payment > 0:
                t_score = _clamp(best_payment / ideal_payment)
            else:
                t_score = 0.5

            vendor_scores[vid] = 0.50 * p_score + 0.30 * d_score + 0.20 * t_score

        # Determine which vendor is objectively better
        sorted_vendors = sorted(vendor_scores.items(), key=lambda x: x[1], reverse=True)
        best_vid = sorted_vendors[0][0]

        # Check if agent negotiated with both vendors (shows good strategy)
        vendor_offers = _collect_vendor_last_offers(history)
        negotiated_both = len(vendor_offers) >= 2

        if chosen_vid == best_vid:
            # Agent picked the better vendor
            base = 1.0
        else:
            # Agent picked the inferior vendor — still gets partial credit
            base = 0.3

        # Bonus for negotiating with both vendors (strategic play)
        if negotiated_both:
            bonus = 0.0  # Already factored into base
        else:
            # Didn't explore both options — small penalty
            base *= 0.8

        return _clamp(base)


# =========================================================================
# Factory functions
# =========================================================================

def get_all_tasks() -> list[Task]:
    """Return all 3 task instances.

    Returns:
        List of [Task1, Task2, Task3] in order of difficulty.
    """
    return [
        Task1_SimplePriceNegotiation(),
        Task2_JobOfferNegotiation(),
        Task3_VendorContractNegotiation(),
    ]


def get_task(task_id: str) -> Task:
    """Return a single task by its task_id.

    Args:
        task_id: One of "task_1", "task_2", "task_3".

    Returns:
        The matching Task instance.

    Raises:
        ValueError: If task_id is not recognized.
    """
    task_map = {t.task_id: t for t in get_all_tasks()}
    if task_id not in task_map:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid IDs: {list(task_map.keys())}"
        )
    return task_map[task_id]
