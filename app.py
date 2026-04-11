from fastapi import FastAPI, Body
import hashlib
import threading

from models import Action, Observation, StepResult, RewardBreakdown
from environment import AutoMindEnv
from tasks import safe_score

app = FastAPI(title="AutoMind OpenEnv Fleet Benchmark", version="1.0.0")

envs: dict[str, AutoMindEnv] = {}
envs_lock = threading.Lock()


def _clamp_all_scores(value):
    """Recursively clamp any field containing 'score' in its key to (0.05, 0.95)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, (int, float)) and "score" in k.lower():
                out[k] = safe_score(v)
            else:
                out[k] = _clamp_all_scores(v)
        return out
    if isinstance(value, list):
        return [_clamp_all_scores(i) for i in value]
    return value


def stable_seed(car_id: str) -> int:
    return int(hashlib.sha256(car_id.encode()).hexdigest()[:8], 16)


def get_env(car_id: str) -> AutoMindEnv:
    with envs_lock:
        if car_id not in envs:
            seed = stable_seed(car_id)
            envs[car_id] = AutoMindEnv(
                seed=seed, vehicle_id=car_id, update_interval_seconds=3
            )
            difficulty = ["easy", "medium", "hard"][seed % 3]
            envs[car_id].reset(task_name="autonomous_control", difficulty=difficulty)
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
    env        = get_env(cid)
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

    # ── Clamp metric scores ───────────────────────────────────────────────────
    if result.metrics:
        result.metrics.safety_score     = safe_score(result.metrics.safety_score)
        result.metrics.efficiency_score = safe_score(result.metrics.efficiency_score)
        result.metrics.diagnosis_score  = safe_score(result.metrics.diagnosis_score)
        result.metrics.sequence_score   = safe_score(result.metrics.sequence_score)

    # ── Clamp info score fields + reward_breakdown ────────────────────────────
    if isinstance(result.info, dict):
        for key in ("task_score", "score"):
            if key in result.info:
                result.info[key] = safe_score(result.info[key])

        breakdown = result.info.get("reward_breakdown")
        if isinstance(breakdown, dict):
            for key in (
                "total", "safety_component", "efficiency_component",
                "diagnosis_component", "service_component",
                "health_component", "sequence_component",
            ):
                if key in breakdown:
                    breakdown[key] = safe_score(breakdown[key])

            # penalty_component is negative — clamp to (-0.95, -0.05)
            if "penalty_component" in breakdown:
                pc = float(breakdown["penalty_component"])
                breakdown["penalty_component"] = max(-0.95, min(-0.05, pc))

    # Final recursive sweep for any remaining score fields
    return _clamp_all_scores(result.model_dump())


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
                "goal": "Identify the active fleet maintenance issue from telemetry",
                "allowed_actions": ["diagnose"],
                "grader": "deterministic fault classifier with partial credit",
            },
            {
                "name": "driving_decision",
                "difficulty": "medium",
                "goal": "Choose the safest immediate roadside maneuver",
                "allowed_actions": [
                    "brake", "accelerate", "turn_left",
                    "turn_right", "continue", "stop",
                ],
                "grader": "deterministic safety policy with partial credit",
            },
            {
                "name": "autonomous_control",
                "difficulty": "hard",
                "goal": "Recover vehicle safely while coordinating service",
                "allowed_actions": [
                    "brake", "accelerate", "turn_left", "turn_right",
                    "continue", "stop", "request_service",
                    "reschedule_service", "cancel_service",
                ],
                "grader": "trajectory score: safety + diagnosis + service + reward shaping",
            },
        ]
    }


@app.get("/ecu/list")
def ecu_list():
    with envs_lock:
        vehicles = [
            {"vin": e.vehicle_profile["vin"], "name": e.vehicle_profile["name"], "car_id": cid}
            for cid, e in envs.items()
        ]
    if not vehicles:
        env = get_env("default")
        vehicles = [
            {"vin": env.vehicle_profile["vin"], "name": env.vehicle_profile["name"], "car_id": "default"}
        ]
    return {"vehicles": vehicles}


@app.get("/ecu/telemetry")
def ecu_telemetry(vin: str | None = None, car_id: str = "default"):
    resolved = car_id
    if vin:
        with envs_lock:
            for cid, env in envs.items():
                if env.vehicle_profile["vin"] == vin:
                    resolved = cid
                    break
    return get_env(resolved).get_ecu_payload()


@app.get("/schema")
def schema():
    return {
        "Observation":     Observation.model_json_schema(),
        "Action":          Action.model_json_schema(),
        "RewardBreakdown": RewardBreakdown.model_json_schema(),
        "StepResult":      StepResult.model_json_schema(),
    }