"""Test all main.py endpoints against the running server."""
import json
import requests

BASE = "http://localhost:7860"

def test(method, path, data=None, label=""):
    url = f"{BASE}{path}"
    r = requests.get(url, timeout=10) if method == "GET" else requests.post(url, json=data, timeout=10)
    status = "PASS" if r.status_code == 200 else f"FAIL({r.status_code})"
    print(f"  [{status}] {method:4s} {path:10s} {label}")
    if r.status_code != 200:
        print(f"         Detail: {r.text[:200]}")
    return r.json() if r.status_code == 200 else None

print("=" * 60)
print("MAIN.PY ENDPOINT TESTS")
print("=" * 60)

# Health endpoints
test("GET", "/", label="root")
test("GET", "/health", label="health")

# Tasks
data = test("GET", "/tasks", label="list tasks")
if data:
    for t in data:
        print(f"    {t['task_id']}: {t['name']} ({t['difficulty']})")

# Reset task_1
print("\n--- Full Episode: Task 1 ---")
data = test("POST", "/reset", {"task_id": "task_1"}, "reset")
if data:
    obs = data["observation"]
    print(f"    Price offer: {obs['counterparty_offer'].get('price')}")
    print(f"    Budget: {obs['agent_constraints'].get('max_budget')}")

# Step propose
data = test("POST", "/step", {"action": {"action_type": "propose", "price": 1000}}, "propose")
if data:
    print(f"    Counter: {data['observation']['counterparty_offer'].get('price'):.0f}")
    print(f"    Reward: {data['reward']}, Done: {data['done']}")

# Step accept
data = test("POST", "/step", {"action": {"action_type": "accept"}}, "accept")
if data:
    print(f"    Reward: {data['reward']}, Done: {data['done']}, Status: {data['observation']['status']}")

# State
data = test("GET", "/state", label="state")
if data:
    s = data["state"]
    print(f"    Keys: {list(s.keys())}")
    print(f"    Status: {s['status']}")

# Grade
data = test("POST", "/grade", {"task_id": "task_1"}, "grade task_1")
if data:
    print(f"    Score: {data['score']}")

# Reset task_3 + quick episode
print("\n--- Full Episode: Task 3 ---")
test("POST", "/reset", {"task_id": "task_3"}, "reset")
test("POST", "/step", {"action": {"action_type": "propose", "vendor_id": "vendor_a", "price": 40000, "delivery_days": 50, "payment_terms_days": 45}}, "propose vendor_a")
test("POST", "/step", {"action": {"action_type": "propose", "vendor_id": "vendor_b", "price": 43000, "delivery_days": 45, "payment_terms_days": 55}}, "propose vendor_b")
data = test("POST", "/step", {"action": {"action_type": "accept"}}, "accept")
if data:
    print(f"    Status: {data['observation']['status']}, Reward: {data['reward']}")

data = test("POST", "/grade", {"task_id": "task_3"}, "grade task_3")
if data:
    print(f"    Score: {data['score']}")

# Error handling
print("\n--- Error Handling ---")
r = requests.post(f"{BASE}/reset", json={"task_id": "task_99"}, timeout=10)
print(f"  [{'PASS' if r.status_code == 400 else 'FAIL'}] POST /reset bad task_id -> {r.status_code}")

r = requests.post(f"{BASE}/grade", json={"task_id": "task_99"}, timeout=10)
print(f"  [{'PASS' if r.status_code == 400 else 'FAIL'}] POST /grade bad task_id -> {r.status_code}")

print("\n" + "=" * 60)
print("ALL MAIN.PY TESTS COMPLETE")
print("=" * 60)
