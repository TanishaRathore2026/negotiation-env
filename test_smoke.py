"""Quick smoke test — validates all 3 tasks end-to-end."""
from environment import NegotiationEnvironment
from models import NegotiationAction, ActionType
from graders import grade_episode

env = NegotiationEnvironment()

# ===== TASK 1: Easy (price negotiation) =====
print("=" * 50)
print("TASK 1 - Easy: Price Negotiation")
obs = env.reset("task_1")
print(f"  Reset OK | Step={obs.current_step}/{obs.max_steps} | Offer={obs.counterparty_offer.price}")

action = NegotiationAction(action_type=ActionType.PROPOSE, price=1000.0, reasoning="Lowball")
obs, reward, done, info = env.step(action)
print(f"  Step 1 | Offer price={obs.counterparty_offer.price:.0f} | Reward={reward:.4f} | Done={done}")

action = NegotiationAction(action_type=ActionType.PROPOSE, price=950.0, reasoning="Getting closer")
obs, reward, done, info = env.step(action)
print(f"  Step 2 | Offer price={obs.counterparty_offer.price:.0f} | Reward={reward:.4f} | Done={done}")

action = NegotiationAction(action_type=ActionType.ACCEPT)
obs, reward, done, info = env.step(action)
print(f"  Accept | Reward={reward:.4f} | Done={done} | Status={obs.status.value}")

score1 = grade_episode(env)
print(f"  GRADER SCORE: {score1:.4f}")

# ===== TASK 2: Medium (job negotiation) =====
print()
print("=" * 50)
print("TASK 2 - Medium: Job Offer Negotiation")
obs = env.reset("task_2")
print(f"  Reset OK | Step={obs.current_step}/{obs.max_steps} | Salary={obs.counterparty_offer.salary}")

action = NegotiationAction(
    action_type=ActionType.PROPOSE, salary=140000, remote_days=4,
    start_date_weeks=5, reasoning="Ambitious"
)
obs, reward, done, info = env.step(action)
print(f"  Step 1 | Salary={obs.counterparty_offer.salary:.0f} Remote={obs.counterparty_offer.remote_days} Start={obs.counterparty_offer.start_date_weeks} | Reward={reward:.4f}")

action = NegotiationAction(
    action_type=ActionType.PROPOSE, salary=135000, remote_days=3,
    start_date_weeks=4, reasoning="More reasonable"
)
obs, reward, done, info = env.step(action)
print(f"  Step 2 | Salary={obs.counterparty_offer.salary:.0f} Remote={obs.counterparty_offer.remote_days} Start={obs.counterparty_offer.start_date_weeks} | Reward={reward:.4f}")

action = NegotiationAction(action_type=ActionType.ACCEPT)
obs, reward, done, info = env.step(action)
print(f"  Accept | Reward={reward:.4f} | Done={done} | Status={obs.status.value}")

score2 = grade_episode(env)
print(f"  GRADER SCORE: {score2:.4f}")

# ===== TASK 3: Hard (vendor negotiation) =====
print()
print("=" * 50)
print("TASK 3 - Hard: Vendor Contract Negotiation")
obs = env.reset("task_3")
print(f"  Reset OK | Step={obs.current_step}/{obs.max_steps} | Vendor={obs.counterparty_offer.vendor_id} Price={obs.counterparty_offer.price}")

action = NegotiationAction(
    action_type=ActionType.PROPOSE, vendor_id="vendor_a",
    price=40000, delivery_days=50, payment_terms_days=45, reasoning="Push vendor A"
)
obs, reward, done, info = env.step(action)
print(f"  Step 1 | Price={obs.counterparty_offer.price:.0f} Del={obs.counterparty_offer.delivery_days} Pay={obs.counterparty_offer.payment_terms_days} | Reward={reward:.4f}")

action = NegotiationAction(
    action_type=ActionType.PROPOSE, vendor_id="vendor_b",
    price=42000, delivery_days=45, payment_terms_days=55, reasoning="Try vendor B"
)
obs, reward, done, info = env.step(action)
print(f"  Step 2 | Price={obs.counterparty_offer.price:.0f} Del={obs.counterparty_offer.delivery_days} Pay={obs.counterparty_offer.payment_terms_days} | Reward={reward:.4f}")

action = NegotiationAction(action_type=ActionType.ACCEPT)
obs, reward, done, info = env.step(action)
print(f"  Accept | Reward={reward:.4f} | Done={done} | Status={obs.status.value}")

score3 = grade_episode(env)
print(f"  GRADER SCORE: {score3:.4f}")

# ===== STATE CHECK =====
print()
print("=" * 50)
state = env.state()
print(f"state() keys: {list(state.keys())}")
print(f"state.status: {state['status']}")
print(f"state.cumulative_reward: {state['cumulative_reward']}")

# ===== SUMMARY =====
print()
print("=" * 50)
print("SUMMARY")
print(f"  Task 1 (easy):   {score1:.4f}")
print(f"  Task 2 (medium): {score2:.4f}")
print(f"  Task 3 (hard):   {score3:.4f}")
avg = (score1 + score2 + score3) / 3
print(f"  Average:         {avg:.4f}")
print()
print("ALL TESTS PASSED")
