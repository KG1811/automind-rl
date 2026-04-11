# ==============================
# AutoMind OpenEnv - Tasks & Graders
# RULE: Every score MUST be strictly within (0.0, 1.0)
# We use MIN=0.05, MAX=0.95 — safe_score() enforces this on every return path
# ==============================

from __future__ import annotations

from typing import Optional

from models import Observation, Action, Metrics

# ---------------------------------------------------------------
# GLOBAL BOUNDS — never 0.0 or 1.0
# ---------------------------------------------------------------
MIN_TASK_SCORE: float = 0.05
MAX_TASK_SCORE: float = 0.95


def safe_score(x: float) -> float:
    """Hard clamp to open interval (0, 1). This is called on EVERY return."""
    return max(MIN_TASK_SCORE, min(MAX_TASK_SCORE, float(x)))


def strict_task_score(score: float) -> float:
    """Round to 3 dp then clamp. Always call this before returning a score."""
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
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
        ],
        "goal": "Choose safest immediate driving action",
    },
    "autonomous_control": {
        "allowed_actions": [
            "brake",
            "accelerate",
            "turn_left",
            "turn_right",
            "continue",
            "stop",
            "request_service",
            "reschedule_service",
            "cancel_service",
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


def grade_fault_diagnosis(action: Optional[Action], observation: Observation) -> float:
    """
    Scores strictly within (0.05, 0.95).
    0.95 = exact match, 0.45 = plausible wrong fault, 0.05 = completely wrong.
    """
    if action is None or action.action_type != "diagnose":
        return MIN_TASK_SCORE  # 0.05

    predicted_fault = (action.reason or "").strip().lower()
    true_fault = detect_true_fault(observation)

    if predicted_fault == true_fault:
        return MAX_TASK_SCORE  # 0.95 — correct

    # Predicted a real fault but wrong one (partial credit)
    if true_fault != "no_fault" and predicted_fault not in ("", "no_fault"):
        return strict_task_score(0.45)

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
    if observation.speed > 90 or (observation.speed > 70 and observation.acceleration > 2.0):
        return "brake"
    if observation.speed > 55:
        return "continue"
    if observation.speed < 40:
        return "accelerate"
    return "continue"


def grade_driving_decision(action: Action, observation: Observation) -> float:
    """
    Deterministic grading with partial credit.
    All return values go through strict_task_score() → strictly (0.05, 0.95).
    """
    action_type = action.action_type
    correct_action = get_safe_action(observation)

    if action_type == correct_action:
        return MAX_TASK_SCORE  # 0.95

    # Near-equivalents — partial credit
    if correct_action == "brake" and action_type == "stop":
        return strict_task_score(0.72)
    if correct_action == "stop" and action_type == "brake":
        return strict_task_score(0.72)
    if correct_action == "continue" and action_type == "accelerate":
        return strict_task_score(0.62)
    if correct_action == "accelerate" and action_type == "continue":
        return strict_task_score(0.62)
    if correct_action == "continue" and action_type == "brake":
        return strict_task_score(0.38)

    # Dangerous action near obstacle
    if observation.distance_to_obstacle < 20 and action_type in ("continue", "accelerate"):
        return MIN_TASK_SCORE  # 0.05

    return MIN_TASK_SCORE  # 0.05


# =====================================
# TASK 3 — FULL AUTONOMOUS CONTROL
# =====================================

def grade_autonomous_control(
    metrics: Metrics,
    info: Optional[dict] = None,
    action: Optional[Action] = None,
) -> float:
    """
    Weighted deterministic score.
    Each metric is already clamped by strict_task_score in environment.py.
    All bonus/penalty magnitudes are sized so the total stays within (0.05, 0.95).
    Final safe_score() call is the absolute guarantee.
    """
    # Base: 0.70 weight total. With metrics in [0.05, 0.95]:
    #   min base = 0.70 * 0.05 = 0.035 → clamped to 0.05
    #   max base = 0.70 * 0.95 = 0.665
    base = (
        0.30 * metrics.safety_score
        + 0.18 * metrics.diagnosis_score
        + 0.12 * metrics.efficiency_score
        + 0.10 * metrics.sequence_score
    )
    score = base

    if info:
        outcome             = info.get("outcome", "")
        alerts              = info.get("alerts", [])
        service_booking     = info.get("service_booking")
        service_recommended = info.get("service_recommended")
        health_score        = float(info.get("health_score", 0))
        collision_risk      = float(info.get("collision_risk", 1.0))
        reward_breakdown    = info.get("reward_breakdown", {})

        severe_alert = any(
            alert in alerts
            for alert in ["ENGINE OVERHEATING", "BRAKE FAILURE", "BATTERY ISSUE", "LOW OIL"]
        )

        # Sub-component contributions — each capped at their metric value * weight
        # max addition here ≈ 0.23 * 0.95 ≈ 0.218
        score += 0.08 * float(reward_breakdown.get("service_component",  MIN_TASK_SCORE))
        score += 0.08 * float(reward_breakdown.get("health_component",   MIN_TASK_SCORE))
        score += 0.04 * float(reward_breakdown.get("safety_component",   MIN_TASK_SCORE))
        score += 0.03 * float(reward_breakdown.get("sequence_component", MIN_TASK_SCORE))

        # Outcome bonuses — kept small enough to not breach 0.95 ceiling
        if outcome == "success_safe_stop":
            score += 0.08
        elif outcome == "episode_timeout" and collision_risk < 0.35 and health_score >= 45:
            score += 0.05
        elif outcome.startswith("failure"):
            score -= 0.12  # penalty, but safe_score() floors at 0.05

        # Service coordination
        if severe_alert and service_booking:
            score += 0.08
        elif (
            severe_alert
            and service_recommended
            and action is not None
            and action.action_type == "request_service"
        ):
            score += 0.05
        elif severe_alert and action is not None and action.action_type == "accelerate":
            score -= 0.07

    # Absolute guarantee — no 0.0 or 1.0 can ever escape
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
    """
    Central entry point.
    ALL paths return a float strictly within (0.05, 0.95).
    """
    if task_name == "fault_diagnosis":
        return grade_fault_diagnosis(action, observation)

    if task_name == "driving_decision":
        if action is None:
            return MIN_TASK_SCORE
        score = grade_driving_decision(action, observation)
        if info and info.get("outcome") == "failure_unsafe_decision":
            score = min(score, strict_task_score(0.45))
        return strict_task_score(score)

    if task_name == "autonomous_control":
        if metrics is None:
            return MIN_TASK_SCORE
        return grade_autonomous_control(metrics, info=info, action=action)

    raise ValueError(f"Unknown task: {task_name}")