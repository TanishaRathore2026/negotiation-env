"""
graders.py — Deterministic grader functions for each task.

Each grader runs a complete episode (reset → step loop → done) and
returns a score in [0.0, 1.0].

Grader scores:
  • 0.0 = complete failure (no agreement, all invalid moves)
  • 0.5 = mediocre deal (agreed but poor terms / wasted turns)
  • 1.0 = perfect negotiation (ideal terms, fast agreement, fair deal)

Used by:
  • The automated evaluation pipeline
  • The baseline inference script
  • The openenv validate pre-submission check
"""

from __future__ import annotations

from typing import Any

from environment import NegotiationEnvironment
from models import NegotiationStatus


def grade_episode(env: NegotiationEnvironment) -> float:
    """Grade a completed episode.

    Reads the environment's internal state and computes a deterministic
    score in [0.0, 1.0] based on:
      • Whether agreement was reached
      • Quality of the deal (progress toward ideal)
      • Efficiency (turns used)
      • Fairness (closeness to ZOPA midpoint)

    Args:
        env: The environment instance *after* the episode has ended
             (i.e. state shows done=True).

    Returns:
        Float score in [0.0, 1.0].
    """
    state = env.state()
    status = state["status"]
    task_id = state["task_id"]
    scenario = state["scenario"]
    history = state["negotiation_history"]
    current_step = state["current_step"]
    max_steps = state["max_steps"]

    # ── No agreement → base score depends on progress made ──
    if status != NegotiationStatus.AGREED.value:
        # Give partial credit for progress even without agreement
        progress_credit = _compute_progress_credit(task_id, scenario, history)
        # Capped at 0.3 — you can't score well without a deal
        return round(min(progress_credit * 0.3, 0.3), 4)

    # ── Agreement reached — compute quality score ──
    deal_quality = _compute_deal_quality(task_id, scenario, history)
    efficiency = _compute_efficiency(current_step, max_steps)
    fairness = _compute_fairness(task_id, scenario, history)

    # Weighted total: deal_quality is most important
    # Weights: deal_quality=0.50, efficiency=0.25, fairness=0.25
    raw_score = (0.50 * deal_quality) + (0.25 * efficiency) + (0.25 * fairness)
    return round(max(0.0, min(1.0, raw_score)), 4)


def grade_task1(env: NegotiationEnvironment) -> float:
    """Convenience: grade a Task 1 (easy) episode."""
    return grade_episode(env)


def grade_task2(env: NegotiationEnvironment) -> float:
    """Convenience: grade a Task 2 (medium) episode."""
    return grade_episode(env)


def grade_task3(env: NegotiationEnvironment) -> float:
    """Convenience: grade a Task 3 (hard) episode."""
    return grade_episode(env)


# =========================================================================
# Internal scoring functions
# =========================================================================

def _compute_progress_credit(
    task_id: str,
    scenario: dict[str, Any],
    history: list[dict[str, Any]],
) -> float:
    """Compute how much progress was made even without agreement.

    Looks at the last counterparty offer and measures how far it moved
    from the opening position toward the agent's ideal.

    Returns:
        Float in [0.0, 1.0].
    """
    last_cp_offer = _get_last_counterparty_offer(history)
    if not last_cp_offer:
        return 0.0

    if task_id == "task_1":
        start = scenario["seller_starting_price"]
        ideal = scenario["buyer_ideal_price"]
        current = last_cp_offer.get("price", start)
        total_range = start - ideal
        if total_range <= 0:
            return 0.5
        achieved = start - current
        return max(0.0, min(1.0, achieved / total_range))

    elif task_id == "task_2":
        issues = scenario["issues"]
        total = 0.0
        weight_sum = 0.0
        for iss in issues:
            name = iss["name"]
            emp_init = iss["employer_initial"]
            cand_ideal = iss["candidate_ideal"]
            current = last_cp_offer.get(name, emp_init)
            full_range = abs(cand_ideal - emp_init)
            if full_range > 0:
                achieved = abs(current - emp_init)
                total += iss["weight"] * min(1.0, achieved / full_range)
            else:
                total += iss["weight"] * 0.5
            weight_sum += iss["weight"]
        return total / weight_sum if weight_sum > 0 else 0.0

    elif task_id == "task_3":
        # Look at progress on the vendor that received the most offers
        vendor_offers = _collect_vendor_last_offers(history)
        if not vendor_offers:
            return 0.0
        
        best_progress = 0.0
        for vid, offer in vendor_offers.items():
            vendor_data = _find_vendor(scenario, vid)
            if not vendor_data:
                continue
            # Price progress
            init_p = vendor_data["initial_price"]
            min_p = vendor_data["min_price"]
            cur_p = offer.get("price", init_p)
            p_range = init_p - min_p
            p_prog = max(0, (init_p - cur_p) / p_range) if p_range > 0 else 0.5

            # Delivery progress
            init_d = vendor_data["initial_delivery_days"]
            min_d = vendor_data["min_delivery_days"]
            cur_d = offer.get("delivery_days", init_d)
            d_range = init_d - min_d
            d_prog = max(0, (init_d - cur_d) / d_range) if d_range > 0 else 0.5

            # Payment progress
            init_t = vendor_data["initial_payment_terms_days"]
            max_t = vendor_data["max_payment_terms_days"]
            cur_t = offer.get("payment_terms_days", init_t)
            t_range = max_t - init_t
            t_prog = max(0, (cur_t - init_t) / t_range) if t_range > 0 else 0.5

            vendor_progress = 0.5 * p_prog + 0.3 * d_prog + 0.2 * t_prog
            best_progress = max(best_progress, vendor_progress)

        return min(1.0, best_progress)

    return 0.0


def _compute_deal_quality(
    task_id: str,
    scenario: dict[str, Any],
    history: list[dict[str, Any]],
) -> float:
    """Compute how good the agreed deal is for the agent.

    Measures distance from the agent's ideal values.

    Returns:
        Float in [0.0, 1.0].
    """
    accepted_offer = _get_accepted_offer(history)
    if not accepted_offer:
        return 0.0

    if task_id == "task_1":
        price = accepted_offer.get("price")
        if price is None:
            return 0.0
        ideal = scenario["buyer_ideal_price"]
        max_budget = scenario["buyer_max_budget"]
        # Best case: price == ideal → 1.0
        # Worst case: price == max_budget → 0.0
        spread = max_budget - ideal
        if spread <= 0:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (price - ideal) / spread))

    elif task_id == "task_2":
        issues = scenario["issues"]
        total = 0.0
        for iss in issues:
            name = iss["name"]
            cand_ideal = iss["candidate_ideal"]
            cand_min = iss["candidate_min"]
            val = accepted_offer.get(name)
            if val is None:
                continue
            spread = abs(cand_ideal - cand_min)
            if spread <= 0:
                total += iss["weight"] * 0.5
                continue
            # How close to ideal?
            dist = abs(val - cand_ideal)
            quality = max(0.0, min(1.0, 1.0 - dist / spread))
            total += iss["weight"] * quality
        return total

    elif task_id == "task_3":
        vid = accepted_offer.get("vendor_id")
        vendor_data = _find_vendor(scenario, vid) if vid else None
        if not vendor_data:
            return 0.0

        budget = scenario["budget"]
        ideal_del = scenario["ideal_delivery_days"]
        ideal_pay = scenario["ideal_payment_terms_days"]

        # Price quality (lower = better)
        price = accepted_offer.get("price", budget)
        min_p = vendor_data["min_price"]
        p_spread = budget - min_p
        p_quality = max(0, 1.0 - (price - min_p) / p_spread) if p_spread > 0 else 0.5

        # Delivery quality (closer to ideal = better)
        delivery = accepted_offer.get("delivery_days", vendor_data["initial_delivery_days"])
        min_d = vendor_data["min_delivery_days"]
        d_spread = abs(vendor_data["initial_delivery_days"] - ideal_del)
        d_quality = max(0, 1.0 - abs(delivery - ideal_del) / d_spread) if d_spread > 0 else 0.5

        # Payment quality (closer to ideal = better)
        payment = accepted_offer.get("payment_terms_days", vendor_data["initial_payment_terms_days"])
        max_t = vendor_data["max_payment_terms_days"]
        t_spread = abs(max_t - vendor_data["initial_payment_terms_days"])
        dist_from_ideal = abs(payment - ideal_pay)
        t_quality = max(0, 1.0 - dist_from_ideal / t_spread) if t_spread > 0 else 0.5

        return 0.5 * p_quality + 0.3 * d_quality + 0.2 * t_quality

    return 0.0


def _compute_efficiency(current_step: int, max_steps: int) -> float:
    """Score efficiency: fewer turns = higher score.

    1.0 = agreed on turn 1, linearly decreasing to 0.2 at max turns.

    Returns:
        Float in [0.0, 1.0].
    """
    if max_steps <= 1:
        return 1.0
    # Linear from 1.0 (turn 1) to 0.2 (max turns)
    return max(0.2, 1.0 - 0.8 * (current_step - 1) / (max_steps - 1))


def _compute_fairness(
    task_id: str,
    scenario: dict[str, Any],
    history: list[dict[str, Any]],
) -> float:
    """Score fairness: how close is the deal to the ZOPA midpoint.

    A perfectly fair deal benefits both parties equally.

    Returns:
        Float in [0.0, 1.0].
    """
    accepted_offer = _get_accepted_offer(history)
    if not accepted_offer:
        return 0.0

    if task_id == "task_1":
        price = accepted_offer.get("price")
        if price is None:
            return 0.0
        seller_min = scenario["seller_min_price"]
        buyer_max = scenario["buyer_max_budget"]
        midpoint = (seller_min + buyer_max) / 2.0
        max_dist = (buyer_max - seller_min) / 2.0
        if max_dist <= 0:
            return 0.5
        dist = abs(price - midpoint)
        return max(0.0, min(1.0, 1.0 - dist / max_dist))

    elif task_id == "task_2":
        issues = scenario["issues"]
        total = 0.0
        for iss in issues:
            name = iss["name"]
            emp_min = iss["employer_min"]
            cand_ideal = iss["candidate_ideal"]
            midpoint = (emp_min + cand_ideal) / 2.0
            max_dist = abs(cand_ideal - emp_min) / 2.0
            val = accepted_offer.get(name)
            if val is None or max_dist <= 0:
                total += iss["weight"] * 0.5
                continue
            dist = abs(val - midpoint)
            total += iss["weight"] * max(0.0, min(1.0, 1.0 - dist / max_dist))
        return total

    elif task_id == "task_3":
        vid = accepted_offer.get("vendor_id")
        vendor_data = _find_vendor(scenario, vid) if vid else None
        if not vendor_data:
            return 0.0

        budget = scenario["budget"]
        min_p = vendor_data["min_price"]
        mid_p = (min_p + budget) / 2.0
        max_p_dist = (budget - min_p) / 2.0

        price = accepted_offer.get("price", budget)
        p_fair = max(0, 1 - abs(price - mid_p) / max_p_dist) if max_p_dist > 0 else 0.5

        min_d = vendor_data["min_delivery_days"]
        ideal_d = scenario["ideal_delivery_days"]
        mid_d = (min_d + ideal_d) / 2.0
        max_d_dist = abs(ideal_d - min_d) / 2.0

        delivery = accepted_offer.get("delivery_days", vendor_data["initial_delivery_days"])
        d_fair = max(0, 1 - abs(delivery - mid_d) / max_d_dist) if max_d_dist > 0 else 0.5

        init_t = vendor_data["initial_payment_terms_days"]
        max_t = vendor_data["max_payment_terms_days"]
        mid_t = (init_t + max_t) / 2.0
        max_t_dist = abs(max_t - init_t) / 2.0

        payment = accepted_offer.get("payment_terms_days", init_t)
        t_fair = max(0, 1 - abs(payment - mid_t) / max_t_dist) if max_t_dist > 0 else 0.5

        return 0.5 * p_fair + 0.3 * d_fair + 0.2 * t_fair

    return 0.0


# =========================================================================
# History helpers
# =========================================================================

def _get_last_counterparty_offer(history: list[dict]) -> dict[str, Any] | None:
    """Get the most recent counterparty offer from history."""
    for entry in reversed(history):
        if entry.get("actor") == "counterparty" and "offer" in entry:
            return entry["offer"]
    return None


def _get_accepted_offer(history: list[dict]) -> dict[str, Any] | None:
    """Get the offer that was accepted (last counterparty offer before accept).
    
    When the agent accepts, they accept the current counterparty offer,
    so we return the most recent counterparty offer from history.
    """
    return _get_last_counterparty_offer(history)


def _collect_vendor_last_offers(history: list[dict]) -> dict[str, dict[str, Any]]:
    """Collect the latest offer from each vendor in the history."""
    vendor_offers: dict[str, dict[str, Any]] = {}
    for entry in history:
        if entry.get("actor") == "counterparty" and "offer" in entry:
            offer = entry["offer"]
            vid = offer.get("vendor_id")
            if vid:
                vendor_offers[vid] = offer
    return vendor_offers


def _find_vendor(scenario: dict[str, Any], vendor_id: str | None) -> dict[str, Any] | None:
    """Find vendor data by vendor_id in the scenario."""
    if not vendor_id:
        return None
    for v in scenario.get("vendors", []):
        if v["vendor_id"] == vendor_id:
            return v
    return None
