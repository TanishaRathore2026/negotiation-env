"""
inference.py — Mandatory baseline inference script for the AI Negotiation
Environment (Meta × Scalar OpenEnv Hackathon).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration — strictly from environment variables (no fallbacks for keys)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ["API_BASE_URL"]          # MUST be set — no default
API_KEY      = os.environ["API_KEY"]               # MUST be set — no default
MODEL        = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# OpenAI client — initialized strictly with injected env vars
# ---------------------------------------------------------------------------

client = None

try:
    from openai import OpenAI
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    print(f"[INFO] OpenAI client initialized. Base URL: {API_BASE_URL}", flush=True)
except Exception as e:
    print(f"[WARN] Could not initialize OpenAI client: {e}", flush=True)
    print("[WARN] Will use fallback heuristic agent.", flush=True)

# ---------------------------------------------------------------------------
# call_env — HTTP helper for environment API
# ---------------------------------------------------------------------------

def call_env(endpoint: str, method: str = "GET", data: dict | None = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data, timeout=60)
        else:
            resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to environment at {url}", flush=True)
        return {}
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out: {method} {url}", flush=True)
        return {}
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}", flush=True)
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error calling env: {e}", flush=True)
        return {}

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TASK1 = """You are a skilled negotiator buying an item. Analyze the current negotiation state and decide your next action.

You must return a JSON object with these exact fields:
{
  "offer": {"price": <float>},
  "message": "<your reasoning>",
  "accept": <true or false>
}

Strategy:
- Start with a reasonable counter-offer below the asking price
- Make gradual concessions toward agreement
- Stay within your budget constraint
- Accept when the price is at or below your budget and close to ideal
- If running low on turns, accept any offer within budget"""

SYSTEM_PROMPT_TASK2 = """You are a skilled negotiator for a job offer. You are negotiating salary, remote days, and start date.

You must return a JSON object with these exact fields:
{
  "offer": {"salary": <float>, "remote_days": <int>, "start_date_weeks": <int>},
  "message": "<your reasoning>",
  "accept": <true or false>
}

Strategy:
- Push for your ideal values but be willing to compromise
- Salary is most important (weight 0.5), then remote days (0.3), then start date (0.2)
- Make trade-offs: concede on less important issues to gain on important ones
- Accept when all issues are at or above your minimum constraints
- If running low on turns, accept any offer meeting your minimums"""

SYSTEM_PROMPT_TASK3 = """You are a skilled procurement negotiator evaluating 2 vendors. You negotiate price, delivery days, and payment terms.

You must return a JSON object with these exact fields:
{
  "offer": {"vendor_id": "<vendor_a or vendor_b>", "price": <float>, "delivery_days": <int>, "payment_terms_days": <int>},
  "message": "<your reasoning>",
  "accept": <true or false>
}

Strategy:
- Negotiate with BOTH vendors to create competition
- Start with the more expensive vendor to push them down
- Then switch to the cheaper vendor for leverage
- Price is most important, then delivery, then payment terms
- Accept when the offer is within budget with reasonable delivery and payment
- If running low on turns, accept the best available deal within budget"""

_SYSTEM_PROMPTS = {
    "task_1": SYSTEM_PROMPT_TASK1,
    "task_2": SYSTEM_PROMPT_TASK2,
    "task_3": SYSTEM_PROMPT_TASK3,
}

# ---------------------------------------------------------------------------
# agent_decide — ask the LLM for the next action (with safe fallback)
# ---------------------------------------------------------------------------

def agent_decide(observation: dict, task_id: str) -> dict:
    if client is None:
        return _fallback_action(observation, task_id)

    system_prompt = _SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPT_TASK1)

    step       = observation.get("current_step", 0)
    max_steps  = observation.get("max_steps", 5)
    remaining  = max_steps - step
    desc       = observation.get("scenario_description", "")
    constraints = observation.get("agent_constraints", {})
    cp_offer   = observation.get("counterparty_offer", {})
    cp_message = observation.get("counterparty_message", "")
    history    = observation.get("negotiation_history", [])

    user_prompt = f"""## Current Negotiation State

Task: {task_id}
Turn: {step} / {max_steps} ({remaining} remaining)

## Scenario
{desc}

## Your Constraints (what you must respect)
{json.dumps(constraints, indent=2)}

## Counterparty's Current Offer
{json.dumps(cp_offer, indent=2)}

## Counterparty's Message
"{cp_message}"

## Recent History (last 4 entries)
{json.dumps(history[-4:], indent=2)}

## IMPORTANT
- You have {remaining} turns left.
- {"THIS IS YOUR LAST TURN. Accept if the offer meets your constraints!" if remaining <= 1 else ""}
- {"Only 2 turns left. Strongly consider accepting if the offer is reasonable." if remaining == 2 else ""}
- Respond with ONLY valid JSON. No markdown, no explanation outside JSON.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
        return _decision_to_action(decision, task_id, observation)
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Using fallback.", flush=True)
        return _fallback_action(observation, task_id)


def _decision_to_action(decision: dict, task_id: str, observation: dict) -> dict:
    accept  = decision.get("accept", False)
    offer   = decision.get("offer", {})
    message = decision.get("message", "")

    if accept:
        return {"action_type": "accept", "reasoning": message}

    action: dict[str, Any] = {"action_type": "propose", "reasoning": message}

    if task_id == "task_1":
        action["price"] = offer.get("price", 0)
    elif task_id == "task_2":
        for k in ["salary", "remote_days", "start_date_weeks"]:
            if k in offer:
                action[k] = offer[k]
    elif task_id == "task_3":
        for k in ["vendor_id", "price", "delivery_days", "payment_terms_days"]:
            if k in offer:
                action[k] = offer[k]

    return action


def _fallback_action(observation: dict, task_id: str) -> dict:
    cp          = observation.get("counterparty_offer", {}) or {}
    constraints = observation.get("agent_constraints", {})
    step        = observation.get("current_step", 0)
    max_steps   = observation.get("max_steps", 5)

    if step >= max_steps - 1:
        return {"action_type": "accept", "reasoning": "Last turn — accepting."}

    if task_id == "task_1":
        cp_price = cp.get("price", 0)
        ideal    = constraints.get("ideal_price", cp_price * 0.7)
        budget   = constraints.get("max_budget", cp_price)
        proposed = min((ideal + cp_price) / 2.0, budget)
        return {"action_type": "propose", "price": round(proposed, 2),
                "reasoning": "Fallback: midpoint of ideal and current."}

    elif task_id == "task_2":
        sal_ideal   = constraints.get("salary_ideal", 140000)
        sal_cp      = cp.get("salary", 120000) or 120000
        rem_ideal   = constraints.get("remote_days_ideal", 4)
        rem_cp      = cp.get("remote_days", 1) or 1
        start_ideal = constraints.get("start_date_weeks_ideal", 5)
        start_cp    = cp.get("start_date_weeks", 2) or 2
        return {"action_type": "propose",
                "salary": round((sal_ideal + sal_cp) / 2.0, 2),
                "remote_days": round((rem_ideal + rem_cp) / 2),
                "start_date_weeks": round((start_ideal + start_cp) / 2),
                "reasoning": "Fallback: midpoint across all issues."}

    elif task_id == "task_3":
        vid      = cp.get("vendor_id", "vendor_a") or "vendor_a"
        cp_price = cp.get("price", 50000) or 50000
        budget   = constraints.get("budget", cp_price)
        ideal_del = constraints.get("ideal_delivery_days", 45)
        cp_del    = cp.get("delivery_days", 90) or 90
        ideal_pay = constraints.get("ideal_payment_terms_days", 60)
        cp_pay    = cp.get("payment_terms_days", 15) or 15
        return {"action_type": "propose", "vendor_id": vid,
                "price": round((budget * 0.75 + cp_price) / 2.0, 2),
                "delivery_days": round((ideal_del + cp_del) / 2),
                "payment_terms_days": round((ideal_pay + cp_pay) / 2),
                "reasoning": "Fallback: midpoint targeting better deal."}

    return {"action_type": "accept", "reasoning": "Fallback: accepting."}

# ---------------------------------------------------------------------------
# run_task — execute a full episode for one task
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> dict:
    try:
        reset_data = call_env("/reset", method="POST", data={"task_id": task_id})
        if not reset_data or "observation" not in reset_data:
            print(f'[START] {{"task_id": "{task_id}", "scenario": "ERROR: reset failed"}}', flush=True)
            print(f'[END] {{"task_id": "{task_id}", "total_reward": 0.0, "steps": 0}}', flush=True)
            return {"task_id": task_id, "total_reward": 0.0, "steps": 0}

        obs = reset_data["observation"]
        scenario_desc  = obs.get("scenario_description", "Unknown")
        scenario_short = scenario_desc[:80].replace('"', '\\"')

        print(f'[START] {{"task_id": "{task_id}", "scenario": "{scenario_short}"}}', flush=True)

        done         = False
        step_num     = 0
        total_reward = 0.0

        while not done:
            action    = agent_decide(obs, task_id)
            step_data = call_env("/step", method="POST", data={"action": action})

            if not step_data or "observation" not in step_data:
                print("[WARN] Step failed, breaking.", flush=True)
                break

            obs          = step_data["observation"]
            reward       = step_data.get("reward", 0.0)
            done         = step_data.get("done", False)
            step_num    += 1
            total_reward += reward

            action_log = _action_summary(action, task_id)
            step_log   = {"step": step_num, "action": action_log,
                          "reward": round(reward, 4), "done": done}
            print(f"[STEP] {json.dumps(step_log)}", flush=True)

        grade_data   = call_env("/grade", method="POST", data={"task_id": task_id})
        grader_score = grade_data.get("score", 0.0) if grade_data else 0.0

        end_log = {"task_id": task_id, "total_reward": round(grader_score, 4), "steps": step_num}
        print(f"[END] {json.dumps(end_log)}", flush=True)
        return end_log

    except Exception as e:
        print(f"[ERROR] run_task({task_id}) crashed: {e}", flush=True)
        print(f'[END] {{"task_id": "{task_id}", "total_reward": 0.0, "steps": 0}}', flush=True)
        return {"task_id": task_id, "total_reward": 0.0, "steps": 0}


def _action_summary(action: dict, task_id: str) -> dict:
    if action.get("action_type") == "accept":
        return {"accept": True}
    summary: dict[str, Any] = {}
    if task_id == "task_1":
        summary["price"] = action.get("price", 0)
    elif task_id == "task_2":
        for k in ["salary", "remote_days", "start_date_weeks"]:
            if k in action:
                summary[k] = action[k]
    elif task_id == "task_3":
        for k in ["vendor_id", "price", "delivery_days", "payment_terms_days"]:
            if k in action:
                summary[k] = action[k]
    return summary

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("AI Negotiation Environment — Baseline Inference", flush=True)
    print(f"  Model : {MODEL}", flush=True)
    print(f"  Env   : {ENV_URL}", flush=True)
    print("=" * 60, flush=True)

    try:
        health = call_env("/health")
        if not health:
            print("[WARN] /health not reachable. Attempting tasks anyway.", flush=True)
        else:
            print(f"  Status: {health.get('status', '?')}", flush=True)
    except Exception as e:
        print(f"[WARN] Health check failed: {e}", flush=True)

    print("-" * 60, flush=True)

    results    = []
    start_time = time.time()

    for task_id in ["task_1", "task_2", "task_3"]:
        print(f"\n>>> Running {task_id}...", flush=True)
        try:
            result = run_task(task_id)
        except Exception as e:
            print(f"[ERROR] {task_id} failed with: {e}", flush=True)
            result = {"task_id": task_id, "total_reward": 0.0, "steps": 0}
        results.append(result)
        time.sleep(2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 60, flush=True)

    task_labels = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}
    total_score = 0.0

    for r in results:
        tid   = r["task_id"]
        score = r["total_reward"]
        steps = r["steps"]
        diff  = task_labels.get(tid, "?")
        print(f"  {tid} ({diff:6s}): score={score:.4f}  steps={steps}", flush=True)
        total_score += score

    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  Average Score : {avg_score:.4f}", flush=True)
    print(f"  Total Time    : {elapsed:.1f}s", flush=True)
    print(f"  Tasks Run     : {len(results)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
