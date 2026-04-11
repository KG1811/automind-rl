
---
title: Automind RL
emoji: "🚑"
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AutoMind OpenEnv

AutoMind predicts vehicle failures and degradation risks before they become critical, then enables the agent to take preventive actions such as raising alerts, choosing safe maneuvers, and scheduling service interventions.

AutoMind OpenEnv is a real-world OpenEnv benchmark for fleet maintenance triage and roadside decision support. Agents must diagnose vehicle issues from telemetry, choose safe immediate maneuvers, and coordinate recovery actions such as service dispatch and rescheduling.

This environment is built around decisions that connected mobility systems and fleet operators actually make in production: interpreting live telemetry, anticipating failures, reacting to degraded vehicle health, and escalating service when safety and maintenance conditions justify it.

## Companion Android App

AutoMind also includes an Android companion app for live vehicle monitoring, predictive alerts, and service workflows.

1. Frontend source code: [AutoMind Android App](https://github.com/lakshitgulia/automind_app_frontend.git)
2. App build / deployed download link: [Download the Android App](https://drive.google.com/file/d/1tdwMznw1ffQzn4oP3eyfBqTWWov77ek8/view?usp=share_link)


## Why This Benchmark Matters

Real operators do not solve game-like tasks. They need systems that can:

- infer the dominant failure mode from noisy telemetry
- choose safe roadside behavior under uncertainty
- decide when roadside support is necessary
- avoid unnecessary dispatch while still reacting quickly to severe faults

AutoMind converts those real operational decisions into a deterministic, typed benchmark suitable for agent evaluation and RL training.

## OpenEnv Compliance

This project includes:

- typed Pydantic `Observation`, `Action`, `Metrics`, and `StepResult` models
- `reset(task_name, difficulty)` support
- `step(action)` returning observation, reward, done, info, and metrics
- `state()` / `/state` support for full current state inspection
- `openenv.yaml` metadata and endpoint definitions
- a working `Dockerfile`
- a root-level `inference.py`

Validation:

```powershell
C:\Users\Khushi\AppData\Roaming\Python\Python313\Scripts\openenv.exe validate
```

## Action Space

Action fields:

- `action_type: str`
- `value: float` strictly within `(0.0, 1.0)`
- `reason: str`

Supported actions:

- `diagnose`
- `brake`
- `accelerate`
- `turn_left`
- `turn_right`
- `continue`
- `stop`
- `request_service`
- `reschedule_service`
- `cancel_service`

## Observation Space

The observation contains vehicle telemetry, fault state, and scenario context, including:

- vehicle dynamics: `speed`, `rpm`, `throttle`, `gear`, `acceleration`
- system load: `engine_load`, `transmission_load`, `fuel_rate`
- health state: `engine_temp`, `oil_level`, `battery_health`
- safety context: `distance_to_obstacle`, `road_condition`, `drive_mode`
- location context: `latitude`, `longitude`, `heading`
- fault flags: `failures`
- recent interaction history: `history`
- expanded telemetry payloads: `vehicle_signals`, `vehicle_events`

## Tasks

### 1. `fault_diagnosis` (`easy`)

Objective:
Identify the dominant active fault from telemetry.

Expected behavior:

- inspect telemetry and fault indicators
- choose `diagnose`
- place the predicted fault label in `reason`

Deterministic grader:

- `0.99` for an exact diagnosis
- `0.50` for a plausible non-primary fault when a real fault exists
- `0.01` otherwise

### 2. `driving_decision` (`medium`)

Objective:
Choose the safest immediate maneuver from the current vehicle state.

Expected behavior:

- balance speed, obstacle distance, and degradation state
- avoid unsafe continuation or acceleration under hazardous conditions

Deterministic grader:

- `0.99` for the safest action
- intermediate partial credit for near-safe alternatives
- `0.01` for clearly unsafe actions


### 3. `autonomous_control` (`hard`)

Objective:
Manage a full recovery episode involving safety, degradation, and service escalation.

Expected behavior:

- respond to evolving collision risk
- handle worsening health conditions
- request or reschedule service when justified
- avoid destructive or unstable action sequences

Deterministic grader:

- combines safety, diagnosis, efficiency, service handling, outcome quality, and sequence quality
- returns a deterministic final score strictly within `(0, 1)`


## Reward Function

AutoMind provides shaped reward over the full trajectory rather than only a binary terminal signal.

Reward components include:

- safety
- efficiency
- diagnosis awareness
- service-decision quality
- health preservation
- sequence quality
- penalties for dangerous actions such as accelerating into severe risk

This gives useful learning signal before the final episode ends.
All task-facing scores and rewards are clamped to remain strictly within `(0, 1)` before being returned by the API.

## Difficulty Progression

- `easy`: mostly straightforward signals and lower ambiguity
- `medium`: tighter tradeoffs between caution and continuity
- `hard`: compound failures, escalation pressure, and longer-horizon control decisions

## Baseline Inference

The root-level `inference.py` uses the OpenAI client and emits structured stdout logs using the required `[START]`, `[STEP]`, and `[END]` markers.

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optional: `ENV_BASE_URL`
- optional: `LOCAL_IMAGE_NAME`

The script reads `HF_TOKEN` for OpenAI-compatible inference and uses the required Meta submission environment variables.

Current baseline scores:

- `fault_diagnosis`: `0.990 / 0.990 / 0.990`
- `driving_decision`: `0.990 / 0.400 / 0.700`
- `autonomous_control`: `0.176 / 0.108 / 0.154`


## Local Setup

```powershell
pip install -r requirements.txt
python pre_submission_check.py
C:\Users\Khushi\AppData\Roaming\Python\Python313\Scripts\openenv.exe validate
docker build .
python inference.py
```

## Deployment

Hugging Face Space:

- root URL: `https://khushi1811-automind-rl.hf.space`
- health check: `https://khushi1811-automind-rl.hf.space/health`
- docs: `https://khushi1811-automind-rl.hf.space/docs`

Primary endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /tasks`
- `GET /schema`

## Repository Guide

- `app.py`: FastAPI server and HTTP endpoints
- `environment.py`: state transitions, reward shaping, and episode logic
- `tasks.py`: deterministic graders
- `models.py`: typed OpenEnv models
- `inference.py`: baseline inference runner

## Summary

AutoMind OpenEnv is a practical automotive benchmark for diagnosis, safe control, and roadside service coordination. It is deployed, validated, containerized, and designed to evaluate whether an agent can make realistic operational decisions rather than solve a toy task.

## Note
This repository is a continuation of my original project and has been cloned for further development and enhancements.
All code has been originally developed by me.
Original Repository: https://github.com/KG1811/RL.git  
This repository contains additional features, improvements and ongoing work submitted for evaluation.
