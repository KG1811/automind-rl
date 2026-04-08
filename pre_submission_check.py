from __future__ import annotations

from typing import Iterable

from fastapi.testclient import TestClient

from app import app
from models import Action


TASKS = [
    ("fault_diagnosis", "easy", Action(action_type="diagnose", value=1.0, reason="no_fault")),
    ("fault_diagnosis", "medium", Action(action_type="diagnose", value=1.0, reason="low_oil")),
    ("fault_diagnosis", "hard", Action(action_type="diagnose", value=1.0, reason="engine_overheating")),
    ("driving_decision", "easy", Action(action_type="accelerate", value=0.5, reason="safe acceleration")),
    ("driving_decision", "medium", Action(action_type="continue", value=0.4, reason="maintain safe trajectory")),
    ("driving_decision", "hard", Action(action_type="brake", value=1.0, reason="highest-priority safety maneuver")),
    ("autonomous_control", "easy", Action(action_type="continue", value=0.4, reason="balanced cruise")),
    ("autonomous_control", "medium", Action(action_type="brake", value=0.6, reason="stabilize vehicle")),
    ("autonomous_control", "hard", Action(action_type="request_service", value=1.0, reason="roadside recovery")),
]


def assert_score_range(value: float, label: str) -> None:
    assert 0.0 <= value <= 1.0, f"{label} out of range: {value}"


def run_checks() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200, "Root endpoint failed"

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200, "/tasks failed"
    tasks_payload = tasks_response.json()["tasks"]
    assert len(tasks_payload) >= 3, "Expected at least 3 tasks"

    seen_task_names = {task["name"] for task in tasks_payload}
    assert {"fault_diagnosis", "driving_decision", "autonomous_control"}.issubset(seen_task_names)

    for index, (task_name, difficulty, action) in enumerate(TASKS, start=1):
        car_id = f"validator-{index}"
        reset_response = client.post(
            "/reset",
            params={"car_id": car_id},
            json={"task_name": task_name, "difficulty": difficulty, "car_id": car_id},
        )
        assert reset_response.status_code == 200, f"/reset failed for {task_name}/{difficulty}"

        state_response = client.get("/state", params={"car_id": car_id})
        assert state_response.status_code == 200, f"/state failed for {task_name}/{difficulty}"
        state_payload = state_response.json()

        assert "observation" in state_payload
        assert "dashboard" in state_payload
        assert "ecu" in state_payload
        assert "info" in state_payload
        assert "ml_predictions" in state_payload
        assert state_payload["info"]["scenario"]["title"]
        assert state_payload["dashboard"]["ml_predictions"]["model_name"] == "automind_predictive_v1"

        step_response = client.post("/step", params={"car_id": car_id}, json=action.model_dump())
        assert step_response.status_code == 200, f"/step failed for {task_name}/{difficulty}"

        step_payload = step_response.json()
        reward = float(step_payload["reward"])
        metrics = step_payload["metrics"]

        assert -1.0 <= reward <= 1.0, f"Reward out of range for {task_name}/{difficulty}: {reward}"
        for metric_name, metric_value in metrics.items():
            assert_score_range(float(metric_value), f"{task_name}/{difficulty}:{metric_name}")

        health_score = int(step_payload["info"]["health_score"])
        assert 0 <= health_score <= 100
        assert "reward_breakdown" in step_payload["info"]
        assert "scenario" in step_payload["info"]
        assert "ml_predictions" in step_payload["info"]

    ecu_list = client.get("/ecu/list")
    assert ecu_list.status_code == 200, "/ecu/list failed"
    vehicles = ecu_list.json()["vehicles"]
    assert vehicles, "No vehicles returned from /ecu/list"

    ecu_payload = client.get("/ecu/telemetry", params={"car_id": "validator-1"})
    assert ecu_payload.status_code == 200, "/ecu/telemetry failed"
    assert "obd" in ecu_payload.json()

    print("pre_submission_check: PASS")


if __name__ == "__main__":
    run_checks()
