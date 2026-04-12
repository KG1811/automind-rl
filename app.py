from fastapi import FastAPI, Body
import requests
import hashlib
import threading
from fastapi.responses import HTMLResponse

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


@app.get("/status")
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
    return _clamp_all_scores(env.get_full_state())


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


@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AutoMind Dashboard</title>

    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a, #020617);
            color: white;
            text-align: center;
        }

        /* Top URL Bar */
        #topBar {
            background: #020617;
            padding: 12px;
            border-bottom: 1px solid #334155;
            font-size: 14px;
        }

        #topBar a {
            color: #22c55e;
            text-decoration: none;
            font-weight: bold;
        }

        #topBar a:hover {
            text-decoration: underline;
        }

        h1 {
            margin-top: 30px;
            font-size: 2.5rem;
        }

        .container {
            margin-top: 40px;
        }

        input {
            padding: 12px;
            width: 300px;
            border-radius: 10px;
            border: none;
            outline: none;
            font-size: 16px;
        }

        button {
            padding: 12px 20px;
            margin: 10px;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }

        .btn-send {
            background: #22c55e;
            color: white;
        }

        .btn-state {
            background: #3b82f6;
            color: white;
        }

        button:hover {
            transform: scale(1.05);
            opacity: 0.9;
        }

        #output {
            margin: 40px auto;
            width: 80%;
            max-width: 900px;
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            text-align: left;
            white-space: pre-wrap;
            font-size: 14px;
            overflow-x: auto;
        }

        .card {
            background: #020617;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            display: inline-block;
        }

        .download {
            margin-top: 15px;
            font-size: 16px;
        }

        .download a {
            color: #38bdf8;
            font-weight: bold;
            text-decoration: none;
        }

        .download a:hover {
            text-decoration: underline;
        }
    </style>
</head>

<body>

    <!-- 🔗 App Info + Download Link -->
    <div id="topBar">
        🔗 <a href="https://drive.google.com/file/d/1tdwMznw1ffQzn4oP3eyfBqTWWov77ek8/view?usp=share_link" target="_blank">
        Download our app on your Android phone
        </a>
    </div>

    <h1>🚗 AutoMind Control Panel</h1>

    <div class="container">
        <div class="card">
            <input id="inputText" type="text" placeholder="Enter action (brake / accelerate / stop)" />
            <br>
            <button class="btn-send" onclick="sendRequest()">Send Action</button>
            <button class="btn-state" onclick="getState()">Get State</button>

            <div class="download">
                📱 <a href="https://drive.google.com/file/d/1tdwMznw1ffQzn4oP3eyfBqTWWov77ek8/view?usp=share_link" target="_blank">
                Download our app on your Android phone
                </a>
            </div>
        </div>
    </div>

    <div id="output">System response will appear here...</div>

    <script>
        async function sendRequest() {
            const text = document.getElementById("inputText").value;

            try {
                const response = await fetch("/step", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        action: text
                    })
                });

                const data = await response.json();
                document.getElementById("output").innerText =
                    JSON.stringify(data, null, 2);

            } catch (error) {
                document.getElementById("output").innerText =
                    "Error: " + error;
            }
        }

        async function getState() {
            try {
                const response = await fetch("/state");
                const data = await response.json();

                document.getElementById("output").innerText =
                    JSON.stringify(data, null, 2);

            } catch (error) {
                document.getElementById("output").innerText =
                    "Error: " + error;
            }
        }
    </script>

</body>
</html>
"""