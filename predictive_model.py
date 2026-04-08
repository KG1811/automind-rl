from __future__ import annotations

import math
from statistics import mean

from models import Observation


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class PredictiveMaintenanceModel:
    """
    Lightweight embedded predictive model.

    This uses logistic-style scoring over telemetry features and short-term trends
    so the simulator can expose forward-looking failure risk instead of only
    threshold-based alarms.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _history_mean(self, observation: Observation, key: str, default: float) -> float:
        values = [
            float(item.state_summary.get(key, default))
            for item in observation.history[-4:]
            if item.state_summary and key in item.state_summary
        ]
        return mean(values) if values else default

    def predict(self, observation: Observation, collision_risk: float) -> dict:
        signals = observation.vehicle_signals
        events = observation.vehicle_events

        mean_temp = self._history_mean(observation, "engine_temp", observation.engine_temp)
        mean_oil = self._history_mean(observation, "oil_level", observation.oil_level)
        mean_battery = self._history_mean(observation, "battery_health", observation.battery_health)
        mean_speed = self._history_mean(observation, "speed", observation.speed)

        temp_trend = observation.engine_temp - mean_temp
        oil_drop_trend = mean_oil - observation.oil_level
        battery_drop_trend = mean_battery - observation.battery_health
        speed_trend = observation.speed - mean_speed

        speed_norm = clamp(observation.speed / 140.0, 0.0, 1.4)
        throttle_norm = clamp(observation.throttle / 100.0, 0.0, 1.0)
        load_norm = clamp(observation.engine_load / 100.0, 0.0, 1.0)
        temp_norm = clamp((observation.engine_temp - 82.0) / 35.0, 0.0, 2.0)
        oil_low_norm = clamp((45.0 - observation.oil_level) / 45.0, 0.0, 1.5)
        voltage_low_norm = clamp((12.6 - signals.battery_voltage) / 1.8, 0.0, 1.6)
        battery_low_norm = clamp((60.0 - observation.battery_health) / 60.0, 0.0, 1.5)
        obstacle_norm = clamp((35.0 - observation.distance_to_obstacle) / 35.0, 0.0, 1.5)
        traction_penalty = 0.22 if observation.road_condition == "wet" else (0.35 if observation.road_condition == "rain" else 0.0)

        overheat_risk = sigmoid(
            -3.0
            + 3.8 * temp_norm
            + 0.9 * load_norm
            + 0.55 * speed_norm
            + 0.20 * throttle_norm
            + 0.35 * clamp(temp_trend / 6.0, 0.0, 1.5)
            + 0.60 * float(observation.failures.engine_overheating)
            + 0.25 * float(events.engine_overheat_warning)
        )

        low_oil_risk = sigmoid(
            -2.8
            + 4.0 * oil_low_norm
            + 0.40 * load_norm
            + 0.35 * clamp(oil_drop_trend / 6.0, 0.0, 1.5)
            + 0.25 * clamp(max(signals.oil_temp - 118.0, 0.0) / 20.0, 0.0, 1.0)
            + 0.30 * float(observation.failures.low_oil)
            + 0.35 * float(events.low_oil_warning)
        )

        battery_risk = sigmoid(
            -3.2
            + 3.4 * voltage_low_norm
            + 2.6 * battery_low_norm
            + 0.40 * clamp(battery_drop_trend / 8.0, 0.0, 1.2)
            + 0.35 * float(observation.failures.battery_issue)
            + 0.25 * float(events.charging_fault)
        )

        brake_risk = sigmoid(
            -3.5
            + 2.6 * obstacle_norm
            + 1.4 * speed_norm
            + 0.65 * float(events.harsh_brake_event)
            + 0.90 * float(observation.failures.brake_failure)
            + traction_penalty
        )

        collision_risk_pred = sigmoid(
            -2.6
            + 3.1 * obstacle_norm
            + 1.5 * speed_norm
            + 0.40 * clamp(speed_trend / 18.0, 0.0, 1.0)
            + 0.55 * float(observation.failures.sensor_failure)
            + 0.60 * float(observation.failures.brake_failure)
            + traction_penalty
            + 1.2 * collision_risk
        )
        collision_risk_pred = max(collision_risk_pred, collision_risk)

        failure_risks = {
            "engine_overheating": round(clamp(overheat_risk, 0.0, 1.0), 3),
            "low_oil": round(clamp(low_oil_risk, 0.0, 1.0), 3),
            "battery_issue": round(clamp(battery_risk, 0.0, 1.0), 3),
            "brake_failure": round(clamp(brake_risk, 0.0, 1.0), 3),
            "collision": round(clamp(collision_risk_pred, 0.0, 1.0), 3),
        }

        primary_failure, primary_risk = max(failure_risks.items(), key=lambda item: item[1])
        confidence = round(clamp(0.58 + (primary_risk * 0.34), 0.58, 0.98), 3)

        if primary_failure == "collision":
            horizon_km = max(1, int(round(observation.distance_to_obstacle / 5.0)))
        else:
            speed_floor = max(12.0, observation.speed)
            horizon_km = max(5, int(round((1.0 - primary_risk) * (speed_floor * 1.8))))

        engine_health = 100.0 - (
            52.0 * failure_risks["engine_overheating"]
            + 22.0 * failure_risks["low_oil"]
            + 0.10 * max(observation.engine_temp - 95.0, 0.0)
        )
        battery_health = 100.0 - (
            58.0 * failure_risks["battery_issue"]
            + 0.55 * max(12.3 - signals.battery_voltage, 0.0) * 20.0
        )
        safety_health = 100.0 - (
            65.0 * failure_risks["collision"]
            + 28.0 * failure_risks["brake_failure"]
        )
        maintenance_health = 100.0 - (
            36.0 * failure_risks["low_oil"]
            + 26.0 * failure_risks["battery_issue"]
            + 22.0 * failure_risks["engine_overheating"]
        )

        engine_health = int(round(clamp(engine_health, 0.0, 100.0)))
        battery_health = int(round(clamp(battery_health, 0.0, 100.0)))
        safety_health = int(round(clamp(safety_health, 0.0, 100.0)))
        maintenance_health = int(round(clamp(maintenance_health, 0.0, 100.0)))
        overall_health = int(round(
            0.38 * engine_health
            + 0.20 * battery_health
            + 0.24 * safety_health
            + 0.18 * maintenance_health
        ))

        return {
            "model_name": "automind_predictive_v1",
            "confidence": confidence,
            "primary_failure": primary_failure,
            "failure_horizon_km": horizon_km,
            "failure_risks": failure_risks,
            "health": {
                "overall": int(clamp(overall_health, 0, 100)),
                "engine": engine_health,
                "battery": battery_health,
                "safety": safety_health,
                "maintenance": maintenance_health,
            },
        }
