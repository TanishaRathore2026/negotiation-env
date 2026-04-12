"""
tasks.py — Task definitions and deterministic graders for the AI Negotiation Environment.
"""

from __future__ import annotations

from typing import Any


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _clamp_score(value: float) -> float:
    """Clamp to strictly open interval (0.001, 0.999) as required by validator."""
    return max(0.001, min(0.999, value))


def _normalize_status(status: Any) -> str:
    if hasattr(status, "value"):
        return str(status.value).lower()
    s = str(status).lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s


def _last_counterparty_offer(history: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in reversed(history):
        if entry.get("actor") == "counterparty" and "offer" in entry:
            return entry["offer"]
    return {}


def _last_agent_offer(history: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in reversed(history):
        if entry.get("actor") == "agent" and "offer" in entry:
            return entry["offer"]
    return {}


def _collect_vendor_last_offers(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    vendor_offers: dict[str, dict[str, Any]] = {}
    for entry in history:
        if entry.get("actor") == "counterparty" and "offer" in entry:
            offer = entry["offer"]
            vid = offer.get("vendor_id")
            if vid:
                vendor_offers[vid] = offer
    return vendor_offers


class Task:
    task_id: str = ""
    name: str = ""
    description: str = ""
    difficulty: str = ""
    max_turns: int = 5
    scenario_id: str = ""

    def grade(self, state: dict[str, Any]) -> float:
        raise NotImplementedError("Subclasses must implement grade()")


class Task1_SimplePriceNegotiation(Task):
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
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)

        seller_start = scenario.get("seller_starting_price", 0)
        seller_min = scenario.get("seller_min_price", 0)
        buyer_max = scenario.get("buyer_max_budget", 0)
        buyer_ideal = scenario.get("buyer_ideal_price", 0)
        zopa_mid = (seller_min + buyer_max) / 2.0

        if status != "agreed":
            last_cp = _last_counterparty_offer(history)
            final_cp_price = last_cp.get("price", seller_start)
            initial_gap = seller_start - buyer_ideal
            current_gap = final_cp_price - buyer_ideal
            if initial_gap <= 0:
                progress = 0.5
            else:
                progress = _clamp(1.0 - (current_gap / initial_gap))
            return round(_clamp_score(progress * 0.3), 4)

        last_cp = _last_counterparty_offer(history)
        final_price = last_cp.get("price", seller_start)

        if final_price > buyer_max:
            return 0.1

        zopa_half_width = (buyer_max - seller_min) / 2.0
        if zopa_half_width > 0:
            distance = abs(final_price - zopa_mid)
            proximity = _clamp(1.0 - (distance / zopa_half_width))
        else:
            proximity = 0.5

        if max_steps > 0:
            efficiency = _clamp(1.0 - (current_step / max_steps) * 0.2)
        else:
            efficiency = 1.0

        budget_range = buyer_max - buyer_ideal
        if budget_range > 0:
            savings = _clamp((buyer_max - final_price) / budget_range)
        else:
            savings = 0.5

        composite = (0.50 * proximity) + (0.25 * savings) + (0.25 * efficiency)
        return round(_clamp_score(composite), 4)


class Task2_JobOfferNegotiation(Task):
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
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)
        issues = scenario.get("issues", [])

        if not issues:
            return 0.001

        if status != "agreed":
            total_progress = 0.0
            total_weight = 0.0
            last_cp = _last_counterparty_offer(history)
            for iss in issues:
                name = iss["name"]
                weight = iss.get("weight", 0.33)
                emp_initial = iss["employer_initial"]
                cand_ideal = iss["candidate_ideal"]
                current_val = last_cp.get(name, emp_initial)
                full_range = abs(cand_ideal - emp_initial)
                if full_range > 0:
                    moved = abs(current_val - emp_initial)
                    issue_progress = _clamp(moved / full_range)
                else:
                    issue_progress = 0.5
                total_progress += weight * issue_progress
                total_weight += weight
            weighted_progress = total_progress / total_weight if total_weight > 0 else 0.0
            return round(_clamp_score(weighted_progress * 0.35), 4)

        last_cp = _last_counterparty_offer(history)
        total_score = 0.0
        total_weight = 0.0

        for iss in issues:
            name = iss["name"]
            weight = iss.get("weight", 0.33)
            cand_ideal = iss["candidate_ideal"]
            cand_min = iss["candidate_min"]
            emp_initial = iss["employer_initial"]
            accepted_val = last_cp.get(name, emp_initial)

            full_range = abs(cand_ideal - emp_initial)
            if full_range > 0:
                achieved = abs(accepted_val - emp_initial)
                issue_quality = _clamp(achieved / full_range)
            else:
                issue_quality = 0.5

            emp_min = iss.get("employer_min", emp_initial)
            zopa_mid = (emp_min + cand_ideal) / 2.0
            zopa_half = abs(cand_ideal - emp_min) / 2.0
            if zopa_half > 0:
                fairness = _clamp(1.0 - abs(accepted_val - zopa_mid) / zopa_half)
            else:
                fairness = 0.5

            issue_score = 0.70 * issue_quality + 0.30 * fairness
            total_score += weight * issue_score
            total_weight += weight

        weighted_score = total_score / total_weight if total_weight > 0 else 0.0
        efficiency_penalty = 0.02 * current_step
        final = weighted_score - efficiency_penalty
        return round(_clamp_score(final), 4)


class Task3_VendorContractNegotiation(Task):
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
        status = _normalize_status(state.get("status", ""))
        scenario = state.get("scenario", {})
        history = state.get("negotiation_history", [])
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.max_turns)
        vendors = scenario.get("vendors", [])

        if not vendors:
            return 0.001

        budget = scenario.get("budget", 0)
        ideal_delivery = scenario.get("ideal_delivery_days", 0)
        ideal_payment = scenario.get("ideal_payment_terms_days", 0)

        if status != "agreed":
            vendor_last_offers = _collect_vendor_last_offers(history)
            if not vendor_last_offers:
                return 0.001
            best_progress = 0.0
            for vid, offer in vendor_last_offers.items():
                v_data = self._find_vendor(vendors, vid)
                if not v_data:
                    continue
                progress = self._vendor_progress(v_data, offer)
                best_progress = max(best_progress, progress)
            return round(_clamp_score(best_progress * 0.25), 4)

        last_cp = _last_counterparty_offer(history)
        chosen_vid = last_cp.get("vendor_id")
        chosen_vendor = self._find_vendor(vendors, chosen_vid)

        if not chosen_vendor:
            return 0.1

        accepted_price = last_cp.get("price", chosen_vendor["initial_price"])
        accepted_delivery = last_cp.get("delivery_days", chosen_vendor["initial_delivery_days"])
        accepted_payment = last_cp.get("payment_terms_days", chosen_vendor["initial_payment_terms_days"])

        min_p = chosen_vendor["min_price"]
        if budget - min_p > 0:
            price_quality = _clamp(1.0 - (accepted_price - min_p) / (budget - min_p))
        else:
            price_quality = 0.5

        init_d = chosen_vendor["initial_delivery_days"]
        min_d = chosen_vendor["min_delivery_days"]
        if init_d - min_d > 0:
            delivery_quality = _clamp((init_d - accepted_delivery) / (init_d - min_d))
        else:
            delivery_quality = 0.5

        init_t = chosen_vendor["initial_payment_terms_days"]
        max_t = chosen_vendor["max_payment_terms_days"]
        if max_t - init_t > 0:
            payment_quality = _clamp((accepted_payment - init_t) / (max_t - init_t))
        else:
            payment_quality = 0.5

        deal_quality = (
            0.50 * price_quality +
            0.30 * delivery_quality +
            0.20 * payment_quality
        )

        if max_steps > 1:
            efficiency = _clamp(1.0 - 0.8 * ((current_step - 1) / (max_steps - 1)))
        else:
            efficiency = 1.0

        vendor_selection = self._score_vendor_selection(
            vendors=vendors,
            chosen_vid=chosen_vid,
            history=history,
            budget=budget,
            ideal_delivery=ideal_delivery,
            ideal_payment=ideal_payment,
        )

        composite = (
            0.40 * deal_quality +
            0.30 * efficiency +
            0.30 * vendor_selection
        )
        return round(_clamp_score(composite), 4)

    @staticmethod
    def _find_vendor(vendors: list[dict[str, Any]], vendor_id: str | None) -> dict[str, Any] | None:
        if not vendor_id:
            return None
        for v in vendors:
            if v.get("vendor_id") == vendor_id:
                return v
        return None

    @staticmethod
    def _vendor_progress(v_data: dict[str, Any], offer: dict[str, Any]) -> float:
        init_p = v_data["initial_price"]
        min_p = v_data["min_price"]
        cur_p = offer.get("price", init_p)
        p_range = init_p - min_p
        p_prog = _clamp((init_p - cur_p) / p_range) if p_range > 0 else 0.5

        init_d = v_data["initial_delivery_days"]
        min_d = v_data["min_delivery_days"]
        cur_d = offer.get("delivery_days", init_d)
        d_range = init_d - min_d
        d_prog = _clamp((init_d - cur_d) / d_range) if d_range > 0 else 0.5

        init_t = v_data["initial_payment_terms_days"]
        max_t = v_data["max_payment_terms_days"]
        cur_t = offer.get("payment_terms_days", init_t)
        t_range = max_t - init_t
        t_prog = _clamp((cur_t - init_t) / t_range) if t_range > 0 else 0.5

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
        if len(vendors) < 2 or not chosen_vid:
            return 0.5

        vendor_scores: dict[str, float] = {}
        for v in vendors:
            vid = v["vendor_id"]
            best_price = v["min_price"]
            best_delivery = v["min_delivery_days"]
            best_payment = v["max_payment_terms_days"]

            p_score = _clamp(1.0 - best_price / budget) if budget > 0 else 0.5
            d_score = _clamp(1.0 - abs(best_delivery - ideal_delivery) / ideal_delivery) if ideal_delivery > 0 else 0.5
            t_score = _clamp(best_payment / ideal_payment) if ideal_payment > 0 else 0.5

            vendor_scores[vid] = 0.50 * p_score + 0.30 * d_score + 0.20 * t_score

        sorted_vendors = sorted(vendor_scores.items(), key=lambda x: x[1], reverse=True)
        best_vid = sorted_vendors[0][0]

        vendor_offers = _collect_vendor_last_offers(history)
        negotiated_both = len(vendor_offers) >= 2

        if chosen_vid == best_vid:
            base = 1.0
        else:
            base = 0.3

        if not negotiated_both:
            base *= 0.8

        return _clamp(base)


def get_all_tasks() -> list[Task]:
    return [
        Task1_SimplePriceNegotiation(),
        Task2_JobOfferNegotiation(),
        Task3_VendorContractNegotiation(),
    ]


def get_task(task_id: str) -> Task:
    task_map = {t.task_id: t for t in get_all_tasks()}
    if task_id not in task_map:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid IDs: {list(task_map.keys())}"
        )
    return task_map[task_id]
