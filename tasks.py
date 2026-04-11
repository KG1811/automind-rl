# ==============================
# AutoMind OpenEnv - Tasks & Graders
# META REQUIREMENT: Every score STRICTLY within (0.0, 1.0)
# 0.0 and 1.0 are INVALID. We use [0.05, 0.95] as hard bounds.
# safe_score() is called on EVERY return path. No exceptions.
# ==============================

from __future__ import annotations

from typing import Optional

from models import Observation, Action, Metrics

# ── Hard bounds ────────────────────────────────────────────────────────────────
MIN_TASK_SCORE: float = 0.05
MAX_TASK_SCORE: float = 0.95


def safe_score(x: float) -> float:
    """Clamp any float to strictly-open (0,1) with 0.05/0.95 margins."""
    return max(MIN_TASK_SCORE, min(MAX_TASK_SCORE, float(x)))


def strict_task_score(score: float) -> float:
    """Round to 3 dp and clamp. Call this on every return value."""
    return float(round(safe_score(score), 3))


# =====================================
# TASK CONFIG
# =====================================

TASK_CONFIG = {
    "fault_diagnosis": {
        "allowed_actions": ["diagnose"],
        "goal": "Identify vehicle fault correctly",
    },
    "driving_decision": {
        "allowed_actions": [
            "brake", "accelerate", "turn_left",
            "turn_right", "continue", "stop",
        ],
        "goal": "Choose safest immediate driving action",
    },
    "autonomous_control": {
        "allowed_actions": [
            "brake", "accelerate", "turn_left", "turn_right",
            "continue", "stop", "request_service",
            "reschedule_service", "cancel_service",
        ],
        "goal": "Full control with safety + diagnosis + efficiency",
    },
}


# =====================================
# TASK 1 — FAULT DIAGNOSIS
# =====================================

def detect_true_fault(observation: Observation) -> str:
    """Deterministic ground-truth fault detection."""
    if observation.engine_temp >= 105:
        return "engine_overheating"
    if observation.oil_level <= 25:
        return "low_oil"
    if observation.battery_health <= 20:
        return "battery_issue"
    return "no_fault"


def grade_fault_diagnosis(
    action: Optional[Action], observation: Observation
) -> float:
    """
    Returns strictly within (0.05, 0.95).
    0.95 = exact match
    0.45 = plausible wrong fault (partial credit)
    0.05 = wrong / no diagnose action
    """
    if action is None or action.action_type != "diagnose":
        return MIN_TASK_SCORE  # 0.05

    predicted = (action.reason or "").strip().lower()
    true_fault = detect_true_fault(observation)

    if predicted == true_fault:
        return MAX_TASK_SCORE  # 0.95

    # Predicted a real fault label, but not the right one
    if true_fault != "no_fault" and predicted not in ("", "no_fault"):
        return strict_task_score(0.45)  # partial credit

    return MIN_TASK_SCORE  # 0.05


# =====================================
# TASK 2 — DRIVING DECISION
# =====================================

def get_safe_action(observation: Observation) -> str:
    """Deterministic safe driving policy."""
    if (
        observation.failures.brake_failure
        or observation.failures.engine_overheating
        or observation.engine_temp > 108
        or observation.oil_level < 24
    ):
        if observation.distance_to_obstacle < 18:
            return "brake"
        return "stop"

    if observation.distance_to_obstacle < 15:
        return "brake"
    if observation.distance_to_obstacle < 28 and observation.speed > 35:
        return "brake"
    if observation.speed > 90 or (
        observation.speed > 70 and observation.acceleration > 2.0
    ):
        return "brake"
    if observation.speed > 55:
        return "continue"
    if observation.speed < 40:
        return "accelerate"
    return "continue"


def grade_driving_decision(action: Action, observation: Observation) -> float:
    """
    Returns strictly within (0.05, 0.95).
    All values go through strict_task_score().
    """
    action_type = action.action_type
    correct = get_safe_action(observation)

    if action_type == correct:
        return MAX_TASK_SCORE  # 0.95

    if correct == "brake" and action_type == "stop":
        return strict_task_score(0.72)
    if correct == "stop" and action_type == "brake":
        return strict_task_score(0.72)
    if correct == "continue" and action_type == "accelerate":
        return strict_task_score(0.62)
    if correct == "accelerate" and action_type == "continue":
        return strict_task_score(0.62)
    if correct == "continue" and action_type == "brake":
        return strict_task_score(0.38)

    if observation.distance_to_obstacle < 20 and action_type in (
        "continue", "accelerate"
    ):
        return MIN_TASK_SCORE

    return MIN_TASK_SCORE  # 0.05


# =====================================
# TASK 3 — AUTONOMOUS CONTROL
# =====================================

def grade_autonomous_control(
    metrics: Metrics,
    info: Optional[dict] = None,
    action: Optional[Action] = None,
) -> float:
    """
    Weighted score. strict_task_score() guarantees output in [0.05, 0.95].
    """
    score = (
        0.30 * metrics.safety_score
        + 0.18 * metrics.diagnosis_score
        + 0.12 * metrics.efficiency_score
        + 0.10 * metrics.sequence_score
    )

    if info:
        outcome             = info.get("outcome", "")
        alerts              = info.get("alerts", [])
        service_booking     = info.get("service_booking")
        service_recommended = info.get("service_recommended")
        health_score        = float(info.get("health_score", 0))
        collision_risk      = float(info.get("collision_risk", 1.0))
        reward_breakdown    = info.get("reward_breakdown", {})

        severe_alert = any(
            a in alerts
            for a in [
                "ENGINE OVERHEATING", "BRAKE FAILURE",
                "BATTERY ISSUE", "LOW OIL",
            ]
        )

        score += 0.08 * float(reward_breakdown.get("service_component",  MIN_TASK_SCORE))
        score += 0.08 * float(reward_breakdown.get("health_component",   MIN_TASK_SCORE))
        score += 0.04 * float(reward_breakdown.get("safety_component",   MIN_TASK_SCORE))
        score += 0.03 * float(reward_breakdown.get("sequence_component", MIN_TASK_SCORE))

        if outcome == "success_safe_stop":
            score += 0.08
        elif (
            outcome == "episode_timeout"
            and collision_risk < 0.35
            and health_score >= 45
        ):
            score += 0.05
        elif outcome.startswith("failure"):
            score -= 0.12

        if severe_alert and service_booking:
            score += 0.08
        elif (
            severe_alert
            and service_recommended
            and action is not None
            and action.action_type == "request_service"
        ):
            score += 0.05
        elif (
            severe_alert
            and action is not None
            and action.action_type == "accelerate"
        ):
            score -= 0.07

    return strict_task_score(score)


# =====================================
# MASTER EVALUATION FUNCTION
# =====================================

def evaluate_task(
    task_name: str,
    action: Optional[Action],
    observation: Observation,
    metrics: Optional[Metrics],
    info: Optional[dict] = None,
) -> float:
    """Always returns float strictly within (0.05, 0.95)."""
    if task_name == "fault_diagnosis":
        return grade_fault_diagnosis(action, observation)

    if task_name == "driving_decision":
        if action is None:
            return MIN_TASK_SCORE
        score = grade_driving_decision(action, observation)
        if info and info.get("outcome") == "failure_unsafe_decision":
            score = min(score, strict_task_score(0.44))
        return strict_task_score(score)

    if task_name == "autonomous_control":
        if metrics is None:
            return MIN_TASK_SCORE
        return grade_autonomous_control(metrics, info=info, action=action)

    raise ValueError(f"Unknown task: {task_name}")