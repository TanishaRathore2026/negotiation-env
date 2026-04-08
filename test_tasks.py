"""Validate tasks.py graders against real environment episodes."""
from environment import NegotiationEnvironment
from models import NegotiationAction, ActionType
from tasks import get_task, get_all_tasks

env = NegotiationEnvironment()

print("=" * 60)
print("TASKS.PY GRADER VALIDATION")
print("=" * 60)

# ── Task 1: Easy — Agreed scenario ──
print("\n--- Task 1: Agreed Scenario ---")
env.reset("task_1", "price_001")
env.step(NegotiationAction(action_type=ActionType.PROPOSE, price=1000.0))
env.step(NegotiationAction(action_type=ActionType.PROPOSE, price=950.0))
env.step(NegotiationAction(action_type=ActionType.ACCEPT))
state = env.state()
t1 = get_task("task_1")
score = t1.grade(state)
print(f"  Status: {state['status']}")
print(f"  Steps: {state['current_step']}/{state['max_steps']}")
print(f"  Grader Score: {score}")
assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
print(f"  PASS (score in [0,1])")

# ── Task 1: Easy — Failed scenario (no accept) ──
print("\n--- Task 1: Failed Scenario (ran out of turns) ---")
env.reset("task_1", "price_001")
for i in range(5):
    env.step(NegotiationAction(action_type=ActionType.PROPOSE, price=800.0))
state = env.state()
score_fail = t1.grade(state)
print(f"  Status: {state['status']}")
print(f"  Steps: {state['current_step']}/{state['max_steps']}")
print(f"  Grader Score: {score_fail}")
assert 0.0 <= score_fail <= 0.3, f"Failed score should be <= 0.3: {score_fail}"
print(f"  PASS (partial credit <= 0.3)")

# ── Task 2: Medium — Agreed scenario ──
print("\n--- Task 2: Agreed Scenario ---")
env.reset("task_2", "job_001")
env.step(NegotiationAction(
    action_type=ActionType.PROPOSE, salary=140000, remote_days=4, start_date_weeks=5
))
env.step(NegotiationAction(
    action_type=ActionType.PROPOSE, salary=135000, remote_days=3, start_date_weeks=4
))
env.step(NegotiationAction(action_type=ActionType.ACCEPT))
state = env.state()
t2 = get_task("task_2")
score2 = t2.grade(state)
print(f"  Status: {state['status']}")
print(f"  Steps: {state['current_step']}/{state['max_steps']}")
print(f"  Grader Score: {score2}")
assert 0.0 <= score2 <= 1.0, f"Score out of range: {score2}"
print(f"  PASS (score in [0,1])")

# ── Task 2: Medium — No agreement ──
print("\n--- Task 2: No Agreement ---")
env.reset("task_2", "job_001")
for i in range(8):
    env.step(NegotiationAction(
        action_type=ActionType.PROPOSE, salary=160000, remote_days=5, start_date_weeks=6
    ))
state = env.state()
score2_fail = t2.grade(state)
print(f"  Status: {state['status']}")
print(f"  Grader Score: {score2_fail}")
assert 0.0 <= score2_fail <= 0.35, f"Failed score should be <= 0.35: {score2_fail}"
print(f"  PASS (partial credit <= 0.35)")

# ── Task 3: Hard — Agreed scenario ──
print("\n--- Task 3: Agreed Scenario ---")
env.reset("task_3", "vendor_001")
env.step(NegotiationAction(
    action_type=ActionType.PROPOSE, vendor_id="vendor_a",
    price=40000, delivery_days=50, payment_terms_days=45
))
env.step(NegotiationAction(
    action_type=ActionType.PROPOSE, vendor_id="vendor_b",
    price=43000, delivery_days=45, payment_terms_days=55
))
env.step(NegotiationAction(action_type=ActionType.ACCEPT))
state = env.state()
t3 = get_task("task_3")
score3 = t3.grade(state)
print(f"  Status: {state['status']}")
print(f"  Steps: {state['current_step']}/{state['max_steps']}")
print(f"  Grader Score: {score3}")
assert 0.0 <= score3 <= 1.0, f"Score out of range: {score3}"
print(f"  PASS (score in [0,1])")

# ── Task 3: Hard — No agreement ──
print("\n--- Task 3: No Agreement ---")
env.reset("task_3", "vendor_001")
for i in range(12):
    env.step(NegotiationAction(
        action_type=ActionType.PROPOSE, vendor_id="vendor_a",
        price=30000, delivery_days=30, payment_terms_days=90
    ))
state = env.state()
score3_fail = t3.grade(state)
print(f"  Status: {state['status']}")
print(f"  Grader Score: {score3_fail}")
assert 0.0 <= score3_fail <= 0.25, f"Failed score should be <= 0.25: {score3_fail}"
print(f"  PASS (partial credit <= 0.25)")

# ── Determinism check ──
print("\n--- Determinism Check ---")
env.reset("task_1", "price_001")
env.step(NegotiationAction(action_type=ActionType.PROPOSE, price=1050.0))
env.step(NegotiationAction(action_type=ActionType.ACCEPT))
state_a = env.state()
score_a = t1.grade(state_a)

# Grade the SAME state again — must produce identical result
score_b = t1.grade(state_a)
score_c = t1.grade(state_a)
print(f"  Run 1: {score_a}")
print(f"  Run 2: {score_b}")
print(f"  Run 3: {score_c}")
assert score_a == score_b == score_c, "DETERMINISM FAILED!"
print(f"  PASS (all 3 runs identical)")

# ── Summary ──
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Task 1 agreed:   {score:.4f}")
print(f"  Task 1 failed:   {score_fail:.4f}")
print(f"  Task 2 agreed:   {score2:.4f}")
print(f"  Task 2 failed:   {score2_fail:.4f}")
print(f"  Task 3 agreed:   {score3:.4f}")
print(f"  Task 3 failed:   {score3_fail:.4f}")
print(f"\n  ALL ASSERTIONS PASSED ✓")
