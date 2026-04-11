from fastapi import FastAPI, Body, Request
import hashlib
from models import Action, Observation, StepResult, RewardBreakdown
from environment import AutoMindEnv
import threading
from tasks import safe_score

app = FastAPI(title="AutoMind OpenEnv Fleet Benchmark", version="1.0.0")

envs: dict[str, AutoMindEnv] = {}
envs_lock = threading.Lock()


def _clamp_score_fields(value):
    """Recursively clamp any dict field containing 'score' to (0.05, 0.95)."""
    if isinstance(value, dict):
        clamped = {}
        for key, item in value.items():
            if isinstance(item, (int, float)) and "score" in key.lower():
                clamped[key] = safe_score(item)
            else:
                clamped[key] = _clamp_score_fields(item)
        return clamped
    if isinstance(value, list):
        return [_clamp_score_fields(item) for item in value]
    return value


def stable_seed(car_id: str) -> int:
    digest = hashlib.sha256(car_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def get_env(car_id: str) -> AutoMindEnv:
    with envs_lock:
        if car_id not in envs:
            seed = stable_seed(car_id)
            envs[car_id] = AutoMindEnv(
                seed=seed,
                vehicle_id=car_id,
                update_interval_seconds=3,
            )
            default_difficulty = ["easy", "medium", "hard"][seed % 3]
            envs[car_id].reset(task_name="autonomous_control", difficulty=default_difficulty)
        return envs[car_id]


@app.get("/")
def root():
    return {
        "status": "AutoMind OpenEnv fleet maintenance benchmark running",
        "active_cars": list(envs.keys()),
    }


@app.get("/health")
def health(car_id: str = "default"):
    env = get_env(car_id)
    return {
        "status": "healthy",
        "initialized": env.is_initialized(),
        "task": env.current_task,
        "difficulty": env.current_difficulty,
        "update_interval_seconds": env.update_interval_seconds,
        "car_id": car_id,
    }


@app.post("/reset")
def reset(payload: dict = Body(default_factory=dict), car_id: str = "default"):
    cid        = payload.get("car_id", car_id)
    task_name  = payload.get("task_name", "fault_diagnosis")
    difficulty = payload.get("difficulty", "easy")

    env = get_env(cid)
    env.update_vehicle_identity(
        display_name=payload.get("vehicle_name"),
        maker=payload.get("vehicle_maker"),
    )
    obs = env.reset(task_name=task_name, difficulty=difficulty)
    return {"observation": obs.model_dump(), "car_id": cid}


@app.post("/step")
def step(action: Action, car_id: str = "default"):
    env = get_env(car_id)
    if not env.is_initialized():
        env.reset()

    result = env.step(action)

    # ── Clamp reward ──────────────────────────────────────────────────────────
    result.reward = safe_score(result.reward)

    # ── Clamp all metric scores ───────────────────────────────────────────────
    if result.metrics:
        result.metrics.safety_score     = safe_score(result.metrics.safety_score)
        result.metrics.efficiency_score = safe_score(result.metrics.efficiency_score)
        result.metrics.diagnosis_score  = safe_score(result.metrics.diagnosis_score)
        result.metrics.sequence_score   = safe_score(result.metrics.sequence_score)

    # ── Clamp info scores + reward_breakdown ──────────────────────────────────
    if result.info:
        for score_key in ("task_score", "score"):
            if score_key in result.info:
                result.info[score_key] = safe_score(result.info[score_key])

        breakdown = result.info.get("reward_breakdown")
        if isinstance(breakdown, dict):
            for key in (
                "total",
                "safety_component",
                "efficiency_component",
                "diagnosis_component",
                "service_component",
                "health_component",
                "sequence_component",
            ):
                if key in breakdown:
                    breakdown[key] = safe_score(breakdown[key])

            # penalty_component is negative — clamp to (-0.95, -0.05)
            if "penalty_component" in breakdown:
                raw = float(breakdown["penalty_component"])
                breakdown["penalty_component"] = max(-0.95, min(-0.05, raw))

    # Final recursive clamp for any leftover score fields
    return _clamp_score_fields(result.model_dump())


@app.get("/state")
def state(car_id: str = "default"):
    env = get_env(car_id)
    if not env.is_initialized():
        env.reset(task_name="autonomous_control", difficulty="medium")
    return env.get_full_state()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "fault_diagnosis",
                "difficulty": "easy",
                "goal": "Identify the active fleet maintenance issue from telemetry before dispatching service",
                "allowed_actions": ["diagnose"],
                "grader": "deterministic fault classifier with partial credit for plausible but non-primary diagnoses",
            },
            {
                "name": "driving_decision",
                "difficulty": "medium",
                "goal": "Choose the safest immediate roadside maneuver from current vehicle context",
                "allowed_actions": ["brake", "accelerate", "turn_left", "turn_right", "continue", "stop"],
                "grader": "deterministic safety policy with partial credit for near-safe actions",
            },
            {
                "name": "autonomous_control",
                "difficulty": "hard",
                "goal": "Recover the vehicle safely while handling overrides and coordinating roadside service",
                "allowed_actions": [
                    "brake", "accelerate", "turn_left", "turn_right",
                    "continue", "stop", "request_service", "reschedule_service", "cancel_service",
                ],
                "grader": "trajectory score combining safety, diagnosis, service escalation, and reward shaping",
            },
        ]
    }


@app.get("/ecu/list")
def ecu_list():
    with envs_lock:
        vehicles = [
            {
                "vin": env.vehicle_profile["vin"],
                "name": env.vehicle_profile["name"],
                "car_id": car_id,
            }
            for car_id, env in envs.items()
        ]
    if not vehicles:
        env = get_env("default")
        vehicles = [
            {
                "vin": env.vehicle_profile["vin"],
                "name": env.vehicle_profile["name"],
                "car_id": "default",
            }
        ]
    return {"vehicles": vehicles}


@app.get("/ecu/telemetry")
def ecu_telemetry(vin: str | None = None, car_id: str = "default"):
    resolved_car_id = car_id
    if vin:
        with envs_lock:
            for existing_car_id, env in envs.items():
                if env.vehicle_profile["vin"] == vin:
                    resolved_car_id = existing_car_id
                    break
    env = get_env(resolved_car_id)
    return env.get_ecu_payload()


@app.get("/schema")
def schema():
    return {
        "Observation":     Observation.model_json_schema(),
        "Action":          Action.model_json_schema(),
        "RewardBreakdown": RewardBreakdown.model_json_schema(),
        "StepResult":      StepResult.model_json_schema(),
    }