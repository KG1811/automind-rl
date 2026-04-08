---
title: Automind RL
emoji: 🚑
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

## OpenEnv Summary

AutoMind OpenEnv is a compact real-world benchmark for fleet maintenance triage and roadside decision support. Agents must diagnose a vehicle issue, choose the safest immediate maneuver, and coordinate recovery actions using live telemetry, predictive maintenance signals, and service escalation.

## Tasks

- `fault_diagnosis` (`easy`): classify the active issue from telemetry
- `driving_decision` (`medium`): choose the safest immediate maneuver
- `autonomous_control` (`hard`): manage recovery, degradation, and service escalation across a full episode

All graders are deterministic and return scores in `[0.0, 1.0]`.

## Spaces

Action fields:
- `action_type`
- `value`
- `reason`

Allowed actions:
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

Observation highlights:
- `speed`, `rpm`, `throttle`, `gear`
- `engine_temp`, `oil_level`, `battery_health`
- `distance_to_obstacle`, `road_condition`, `drive_mode`
- `latitude`, `longitude`, `heading`
- `failures`, `history`

## Local Validation

```powershell
python pre_submission_check.py
C:\Users\Khushi\AppData\Roaming\Python\Python313\Scripts\openenv.exe validate
docker build .
python inference.py
```

## Current Baseline

- `fault_diagnosis`: `1.000 / 1.000 / 1.000`
- `driving_decision`: `1.000 / 0.400 / 0.700`
- `autonomous_control`: `0.176 / 0.108 / 0.154`

# 🚑 Automind RL

Reinforcement learning–based system for intelligent real-time decision making and routing optimization.
