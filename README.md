<<<<<<< HEAD
# 🤝 AI Negotiation Environment

## Overview
This environment simulates complex, multi-issue negotiations between an AI agent and a programmatic or human opponent. 
It matters for agent training because it requires agents to balance competing objectives, perform multi-step planning, handle natural language reasoning, and adapt to hidden constraints from the opposing party.
"Agents learn to negotiate like humans do"

## Environment Description
Negotiation is a fundamental human activity that extends far beyond simple bartering. Whether it's agreeing on a salary, striking up a vendor contract, or settling a dispute, real-world negotiations are complex, multi-turn interactions where parties hold private information and competing interests.

What makes this unique as an RL environment is its structured yet open-ended nature. Agents must communicate their reasoning via natural language, submit structured numerical offers, and decide whether to accept ongoing proposals—all within a strict turn limit. 

The negotiation loop works via a sequence of turns where the agent evaluates the current state, formulates reasoning, and submits a counteroffer or an acceptance. The opponent responds programmatically based on predefined constraints. The episode terminates when an agreement is reached, maximum turns are exhausted, or an invalid move is made.
---
title: Negotiation Env
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
port: 7860
tags:
  - openenv
---
## Action Space

| Field | Type | Description |
| :--- | :--- | :--- |
| `offer` | dict[str, float] | Values for each negotiation issue |
| `message` | str | Agent reasoning |
| `accept` | bool | Whether agent accepts current offer |

## Observation Space  

| Field | Type | Description |
| :--- | :--- | :--- |
| `turn` | int | Current turn number |
| `max_turns` | int | Maximum allowed turns |
| `current_offer` | dict[str, float] | Opponent's current offer |
| `opponent_last_offer` | dict[str, float] | Opponent's previous offer |
| `context` | str | Scenario description |
| `issues` | list[str] | Negotiation issues |
| `constraints` | dict[str, float] | Agent's own constraints |

## Reward Function

The reward function evaluates the overall session and returns a score between [0.0, 1.0]. It consists of four main elements:
- **progress_reward**: measures closeness to agreement each step
- **efficiency_penalty**: -0.02 per turn (encourages quick deals)
- **fairness_bonus**: rewards deals near ZOPA midpoint
- **failure_penalty**: punishes invalid moves

## Tasks

### Task 1 — Simple Price Negotiation (Easy)
- **Description**: Negotiate price of a used laptop as a buyer.
- **Objective**: Agree on a price strictly below the buyer's maximum budget but high enough for the seller to accept.
- **Grader Logic**: The opponent is a rule-based seller that will linearly concede down to its secret reserve price. If the agent offers below the reserve, it is rejected.
- **Expected Score Range**: 0.70 - 0.90 (when successful)

### Task 2 — Job Offer Negotiation (Medium)
- **Description**: Negotiate salary, remote days, and start date.
- **Objective**: Find an optimal trade-off matching both your own requirements and the hiring manager's constraints.
- **Grader Logic**: The opponent calculates a weighted sum of the 3 issues and will only accept if the overall proposed offer beats their internal threshold. They will concede gradually turn-by-turn.
- **Expected Score Range**: 0.50 - 0.80

### Task 3 — Vendor Contract Negotiation (Hard)
- **Description**: Negotiate with 2 vendors on price, delivery, payment terms.
- **Objective**: Manage a tougher opponent representing multiple competitive issues.
- **Grader Logic**: Opponent employs non-linear negotiation tactics, holding firm on price but offering flexibility on payment terms. Strong anchoring and creative proposals are required. 
- **Expected Score Range**: 0.30 - 0.60

## Setup & Usage

### Local Setup
step by step commands:
```bash
git clone https://github.com/your-username/negotiation-env.git
cd negotiation-env
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t negotiation-env .
docker run -p 7860:7860 negotiation-env
```

### Run Baseline Inference
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

## Baseline Scores

| Task | Difficulty | Baseline Score |
| :--- | :--- | :--- |
| task_1 | Easy | ~0.72 |
| task_2 | Medium | ~0.58 |
| task_3 | Hard | ~0.41 |

## API Reference

### GET /state
Returns general environment description and metadata.
```bash
curl -X GET http://localhost:7860/state
```

### GET /tasks
Returns a list of available tasks in the environment.
```bash
curl -X GET http://localhost:7860/tasks
```

### POST /reset
Initializes an environment task and returns the initial observation state.
```bash
curl -X POST http://localhost:7860/reset \
-H "Content-Type: application/json" \
-d '{"task_id": "task_1", "seed": 42}'
```

### POST /step
Submits the agent's action and advances the environment. 
```bash
curl -X POST http://localhost:7860/step \
-H "Content-Type: application/json" \
-d '{"action": {"offer": {"price": 950}, "message": "I can only do 950", "accept": false}}'
```

### POST /grade
Terminates the active episode (if it isn't already) and retrieves the final score. 
```bash
curl -X POST http://localhost:7860/grade
```

## Environment Variables

| Variable | Description |
| :--- | :--- |
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | Hugging Face token |
=======
---
title: Tan12space
emoji: 👁
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> c1396f95751c16c0538d11a02c8dfa577c212699
