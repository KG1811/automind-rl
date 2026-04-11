from __future__ import annotations

import hashlib
import random
import time
from typing import Any
from typing import Optional

from models import (
    Observation,
    FailureState,
    EpisodeState,
    Action,
    StepResult,
    Metrics,
    RewardBreakdown,
    TelemetryState,
)
from simulator import AutoMindSimulator
from service_engine import find_nearest_service, build_service_schedule
from tasks import grade_driving_decision, grade_fault_diagnosis, strict_task_score
from vehicle_payload import build_vehicle_events, build_vehicle_signals
from predictive_model import PredictiveMaintenanceModel


class AutoMindEnv:
    def __init__(
        self,
        seed: int = 42,
        max_steps: int = 20,
        update_interval_seconds: int = 10,
        vehicle_id: str = "default",
    ) -> None:
        self.seed = seed
        self.vehicle_id = vehicle_id
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.update_interval_seconds = update_interval_seconds

        self.simulator = AutoMindSimulator(seed=seed, dt_seconds=float(update_interval_seconds))
        self.predictive_model = PredictiveMaintenanceModel(seed=seed)
        self.vehicle_profile = self._build_vehicle_profile()

        self.current_task: Optional[str] = None
        self.current_difficulty: Optional[str] = None

        self.episode_state: EpisodeState = EpisodeState(max_steps=max_steps)
        self.current_observation: Optional[Observation] = None
        self.current_true_state: Optional[TelemetryState] = None

        self.last_action = Action(action_type="continue", value=0.25, reason="background cruise")
        self.background_action = Action(action_type="continue", value=0.25, reason="background cruise")
        self.override_active = False
        self.override_count = 0
        self.last_background_sync_at = time.monotonic()
        self.last_service_booking: Optional[dict] = None
        self.last_service_recommended: Optional[dict] = None
        self.pending_service_schedule_request: Optional[dict[str, str]] = None
        self.service_autobook_suppressed = False
        self.sequence_id = 0
        self.current_fault_phase = self.vehicle_profile["fault_profile"]
        self.current_scenario: dict = {}
        self.last_reward_breakdown = RewardBreakdown(
            total=strict_task_score(0),
            safety_component=strict_task_score(1),
            efficiency_component=strict_task_score(1),
            diagnosis_component=strict_task_score(0.5),
            service_component=strict_task_score(0),
            health_component=strict_task_score(1),
            sequence_component=strict_task_score(0),
            penalty_component=-1e-2,
        )

        self.last_metrics = Metrics(
            safety_score=strict_task_score(1),
            efficiency_score=strict_task_score(1),
            diagnosis_score=strict_task_score(0.5),
            sequence_score=strict_task_score(0),
        )
        self.last_reward = strict_task_score(0)
        self.last_done = False
        self.last_info: dict = {
            "outcome": "not_initialized",
            "collision_risk": 0,
            "override_active": False,
            "override_count": 0,
            "step_count": 0,
            "health_score": 100,
            "alerts": [],
            "active_alerts": [],
            "subsystem_health": {},
            "service_recommended": None,
            "service_booking": None,
            "reward_breakdown": self.last_reward_breakdown.model_dump(),
            "scenario": {},
            "task_score": strict_task_score(0),
            "score": strict_task_score(0),
        }

    def _strict_metric_score(self, value: float) -> float:
        return strict_task_score(value)

    def _build_vehicle_profile(self) -> dict:
        catalog = [
            {"name": "Creta SX 2021", "maker": "Hyundai"},
            {"name": "Brezza ZXi 2025", "maker": "Maruti Suzuki"},
            {"name": "Punch Creative 2024", "maker": "Tata"},
            {"name": "Scorpio N Z8 2023", "maker": "Mahindra"},
            {"name": "i20 Asta 2022", "maker": "Hyundai"},
            {"name": "WagonR VXi 2020", "maker": "Maruti Suzuki"},
        ]
        fault_profiles = [
            "engine",
            "battery",
            "oil",
            "safety",
        ]
        idx = self.seed % len(catalog)
        suffix = hashlib.sha1(self.vehicle_id.encode("utf-8")).hexdigest()[:8].upper()
        return {
            "vin": f"AM{suffix}",
            "name": catalog[idx]["name"],
            "maker": catalog[idx]["maker"],
            "fault_profile": fault_profiles[self.seed % len(fault_profiles)],
        }

    def update_vehicle_identity(
        self,
        display_name: Optional[str] = None,
        maker: Optional[str] = None,
    ) -> None:
        if display_name:
            self.vehicle_profile["name"] = display_name.strip()
        if maker:
            self.vehicle_profile["maker"] = maker.strip()

    def _parse_service_schedule_request(self, action_reason: str) -> Optional[dict[str, str]]:
        if not action_reason:
            return None

        request: dict[str, str] = {}
        for chunk in action_reason.split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            normalized = key.strip().lower()
            cleaned = value.strip()
            if normalized in {"requested_date", "requested_time"} and cleaned:
                request[normalized] = cleaned

        return request or None

    def _difficulty_rng(self, difficulty: str) -> random.Random:
        difficulty_offsets = {"easy": 11, "medium": 29, "hard": 47}
        return random.Random((self.seed * 1009) + difficulty_offsets[difficulty])

    def _gear_from_speed(self, speed: float) -> int:
        if speed < 1:
            return 0
        if speed < 20:
            return 1
        if speed < 40:
            return 2
        if speed < 60:
            return 3
        if speed < 85:
            return 4
        if speed < 120:
            return 5
        return 6

    def _drive_mode_from_speed(self, speed: float, throttle: float) -> str:
        if speed < 1:
            return "idle"
        if throttle > 65:
            return "sport"
        if speed < 35:
            return "city"
        return "cruise"

    def _spawn_location(self, rng: random.Random) -> tuple[float, float, float]:
        start_points = [
            (28.613900, 77.209000, 0),
            (28.459500, 77.026600, 82.0),
            (28.535500, 77.391000, 48.0),
            (28.669200, 77.453800, 124.0),
            (28.408900, 77.317800, 201.0),
        ]
        lat, lon, heading = start_points[self.seed % len(start_points)]
        return (
            round(lat + rng.uniform(-0.01, 1e-2), 6),
            round(lon + rng.uniform(-0.01, 1e-2), 6),
            round((heading + rng.uniform(-18.0, 18.0)) % 360.0, 2),
        )

    def _compute_service_due(self, odometer_km: float) -> dict:
        interval_km = 15000
        next_due = int(((odometer_km // interval_km) + 1) * interval_km)
        remaining_km = max(0, int(round(next_due - odometer_km)))
        return {
            "interval_km": interval_km,
            "next_due_km": next_due,
            "remaining_km": remaining_km,
            "service_due_now": remaining_km <= 1200,
        }

    def _predictive_snapshot(self, obs: Observation, collision_risk: float) -> dict:
        return self.predictive_model.predict(obs, collision_risk)

    def _compute_range_km(self, obs: Observation) -> int:
        signals = obs.vehicle_signals
        fuel_liters = 50.0 * (signals.fuel_level / 100.0)
        speed = max(5.0, signals.speed)
        liters_per_100km = max(4.8, min(18.0, (signals.fuel_rate / speed) * 100.0))

        if signals.drive_mode == "sport":
            liters_per_100km *= 1.12
        elif signals.drive_mode == "city":
            liters_per_100km *= 1.06
        elif signals.drive_mode == "idle":
            liters_per_100km *= 1.35

        range_km = (fuel_liters / liters_per_100km) * 100.0
        return max(0, int(round(range_km)))

    def _compute_health_snapshot(self, obs: Observation, collision_risk: float) -> dict:
        signals = obs.vehicle_signals
        events = obs.vehicle_events
        prediction = self._predictive_snapshot(obs, collision_risk)
        predicted_health = prediction["health"]
        risks = prediction["failure_risks"]

        engine_health = float(predicted_health["engine"])
        battery_health_score = float(predicted_health["battery"])
        safety_health = float(predicted_health["safety"])
        maintenance_health = float(predicted_health["maintenance"])

        if signals.oil_temp >= 135:
            engine_health -= 10
        elif signals.oil_temp >= 120:
            engine_health -= 5

        if events.charging_fault:
            battery_health_score -= 10
        if events.overspeed_event:
            safety_health -= 8
        if obs.failures.sensor_failure:
            safety_health -= 10

        service_due = self._compute_service_due(signals.odometer_km)
        if service_due["service_due_now"]:
            maintenance_health -= 14
        elif service_due["remaining_km"] < 3000:
            maintenance_health -= 6
        if events.dtc_count >= 3:
            maintenance_health -= 10
        elif events.dtc_count >= 1:
            maintenance_health -= 4

        engine_health = max(0, min(100, int(round(engine_health))))
        battery_health_score = max(0, min(100, int(round(battery_health_score))))
        safety_health = max(0, min(100, int(round(safety_health))))
        maintenance_health = max(0, min(100, int(round(maintenance_health))))

        overall = int(round(
            (0.38 * engine_health)
            + (0.20 * battery_health_score)
            + (0.24 * safety_health)
            + (0.18 * maintenance_health)
        ))

        return {
            "overall": max(0, min(100, overall)),
            "engine": engine_health,
            "battery": battery_health_score,
            "safety": safety_health,
            "maintenance": maintenance_health,
            "service_due": service_due,
            "ml_predictions": prediction,
            "predicted_failure": prediction["primary_failure"],
            "predicted_failure_risk": risks[prediction["primary_failure"]],
        }

    def _build_active_alerts(self, obs: Observation, collision_risk: float) -> list[dict]:
        signals = obs.vehicle_signals
        events = obs.vehicle_events
        prediction = self._predictive_snapshot(obs, collision_risk)
        risks = prediction["failure_risks"]
        alerts: list[dict] = []

        if risks["engine_overheating"] >= 0.58:
            alerts.append({
                "code": "ENGINE_OVERHEATING",
                "severity": "CRITICAL" if risks["engine_overheating"] >= 0.78 or obs.engine_temp >= 115 else "WARN",
                "title": "Predicted Engine Overheating",
                "message": (
                    f"Predictive model estimates {int(round(risks['engine_overheating'] * 100))}% "
                    f"overheating risk within {prediction['failure_horizon_km']} km."
                ),
                "component": "engine",
            })

        if risks["low_oil"] >= 0.55:
            alerts.append({
                "code": "LOW_OIL",
                "severity": "CRITICAL" if risks["low_oil"] >= 0.8 or obs.oil_level <= 15 else "WARN",
                "title": "Predicted Oil System Failure",
                "message": (
                    f"Predictive model estimates {int(round(risks['low_oil'] * 100))}% "
                    f"oil-system risk. Current oil level is {obs.oil_level:.1f}%."
                ),
                "component": "engine",
            })

        if risks["battery_issue"] >= 0.52:
            alerts.append({
                "code": "BATTERY_ISSUE",
                "severity": "CRITICAL" if risks["battery_issue"] >= 0.78 or signals.battery_voltage < 11.2 else "WARN",
                "title": "Predicted Battery Failure",
                "message": (
                    f"Predictive model estimates {int(round(risks['battery_issue'] * 100))}% "
                    f"battery failure risk. Voltage is {signals.battery_voltage:.2f}V."
                ),
                "component": "battery",
            })

        if risks["brake_failure"] >= 0.5 or obs.failures.brake_failure:
            alerts.append({
                "code": "BRAKE_FAILURE",
                "severity": "CRITICAL" if risks["brake_failure"] >= 0.72 or obs.failures.brake_failure else "WARN",
                "title": "Predicted Brake Degradation",
                "message": (
                    f"Predictive model estimates {int(round(risks['brake_failure'] * 100))}% "
                    "brake system degradation risk."
                ),
                "component": "safety",
            })

        if obs.failures.sensor_failure:
            alerts.append({
                "code": "SENSOR_FAILURE",
                "severity": "WARN",
                "title": "Sensor Array Fault",
                "message": "One or more sensors are reporting degraded confidence.",
                "component": "diagnostics",
            })

        if risks["collision"] >= 0.55:
            alerts.append({
                "code": "COLLISION_RISK",
                "severity": "CRITICAL" if risks["collision"] >= 0.82 else "WARN",
                "title": "Elevated Collision Risk",
                "message": f"Predicted forward collision risk is {int(round(risks['collision'] * 100))}%.",
                "component": "safety",
            })

        service_due = self._compute_service_due(signals.odometer_km)
        if service_due["service_due_now"]:
            alerts.append({
                "code": "SERVICE_DUE",
                "severity": "INFO",
                "title": "Service Recommended",
                "message": f"Routine service is due within {service_due['remaining_km']} km.",
                "component": "maintenance",
            })

        return alerts

    def _gear_display(self, gear: int) -> str:
        if gear <= 0:
            return "P"
        return f"D{gear} Auto"

    def _drive_mode_display(self, drive_mode: str) -> str:
        return {
            "idle": "Parked",
            "city": "City",
            "cruise": "Cruise",
            "sport": "Sport",
        }.get(drive_mode, "Cruise")

    def _engine_status(self, engine_health: int) -> str:
        if engine_health >= 85:
            return "NORMAL"
        if engine_health >= 60:
            return "ATTENTION"
        return "CRITICAL"

    def _build_dashboard_payload(
        self,
        observation: Observation,
        collision_risk: float,
        health_snapshot: dict,
        active_alerts: list[dict],
        service_recommended: Optional[dict],
        service_booking: Optional[dict],
    ) -> dict:
        signals = observation.vehicle_signals
        service_due = health_snapshot["service_due"]
        range_km = self._compute_range_km(observation)
        driving_safety_score = max(0, min(100, int(round(100 - (collision_risk * 100)))))

        return {
            "vehicle": {
                "car_id": self.vehicle_id,
                "vin": self.vehicle_profile["vin"],
                "name": self.vehicle_profile["name"],
                "maker": self.vehicle_profile["maker"],
                "status": "OFFLINE" if not signals.ignition_on else "ONLINE",
            },
            "quick_telemetry": {
                "speed_kmph": round(observation.speed, 1),
                "engine_temp_c": round(observation.engine_temp, 1),
                "battery_pct": int(round(observation.battery_health)),
                "battery_v": round(signals.battery_voltage, 2),
            },
            "trip": {
                "drive_mode": observation.drive_mode,
                "drive_mode_display": self._drive_mode_display(observation.drive_mode),
                "range_km": range_km,
                "fuel_level_pct": round(signals.fuel_level, 1),
                "odometer_km": round(signals.odometer_km, 1),
                "gear_display": self._gear_display(observation.gear),
            },
            "health": {
                "overall_score": health_snapshot["overall"],
                "status": "GOOD" if health_snapshot["overall"] >= 80 else ("ATTENTION" if health_snapshot["overall"] >= 55 else "CRITICAL"),
                "engine_health": health_snapshot["engine"],
                "battery_health": health_snapshot["battery"],
                "safety_health": health_snapshot["safety"],
                "maintenance_health": health_snapshot["maintenance"],
                "engine_status": self._engine_status(health_snapshot["engine"]),
                "predicted_failure": health_snapshot["predicted_failure"],
                "predicted_failure_risk_pct": int(round(health_snapshot["predicted_failure_risk"] * 100)),
            },
            "safety": {
                "driving_safety_score": driving_safety_score,
                "collision_risk_pct": int(round(collision_risk * 100)),
                "distance_to_obstacle_m": round(observation.distance_to_obstacle, 1),
            },
            "ml_predictions": health_snapshot["ml_predictions"],
            "maintenance": {
                "service_due_now": service_due["service_due_now"],
                "remaining_km": service_due["remaining_km"],
                "next_due_km": service_due["next_due_km"],
                "service_recommended": service_recommended,
                "service_booking": service_booking,
            },
            "active_alerts": active_alerts,
        }

    def _build_ecu_snapshot(
        self,
        observation: Observation,
        collision_risk: float,
        active_alerts: list[dict],
    ) -> dict:
        signals = observation.vehicle_signals
        events = observation.vehicle_events
        prediction = self._predictive_snapshot(observation, collision_risk)
        parked = events.parked
        battery_v = signals.battery_voltage

        return {
            "vin": self.vehicle_profile["vin"],
            "vehicle_name": self.vehicle_profile["name"],
            "request_id": "STATE",
            "ts_ms": int(time.time() * 1000),
            "mode": observation.drive_mode,
            "seq": self.sequence_id,
            "obd": {
                "rpm": int(round(observation.rpm)),
                "speed_kmph": int(round(observation.speed)),
                "coolant_c": int(round(observation.engine_temp)),
                "oil_temp_c": int(round(signals.oil_temp)),
                "oil_pressure_kpa": int(round(signals.oil_pressure)),
                "throttle_pct": int(round(observation.throttle)),
                "battery_v": round(battery_v, 2),
                "fuel_level_pct": int(round(signals.fuel_level)),
                "dtc_count": events.dtc_count,
                "ignition_status": int(signals.ignition_on),
                "charging_status": int(signals.charging_active),
                "odometer_km": round(signals.odometer_km, 2),
                "mil_status": int(events.mil_status),
                "gear_position": self._gear_display(observation.gear),
            },
            "vibration": {
                "engine_rms": round(0.08 + (observation.rpm / 5200.0) * 0.7, 3),
                "wheel_rms": round(0.05 + (observation.speed / 160.0) * 0.35, 3),
            },
            "health_flags": {
                "misfire_risk": int(observation.rpm > 3600 and observation.throttle < 25),
                "overheat_risk": int(prediction["failure_risks"]["engine_overheating"] >= 0.58),
                "low_oil_risk": int(prediction["failure_risks"]["low_oil"] >= 0.55),
            },
            "safety": {
                "crash_detected": int(collision_risk >= 0.97),
                "collision_risk_pct": int(round(collision_risk * 100)),
            },
            "ml_predictions": prediction,
            "power": {
                "battery_disconnected": int(events.battery_disconnect_event),
                "parked": int(parked),
                "low_battery_warn": int(11.2 < battery_v < 12.2),
                "low_battery_crit": int(0.5 < battery_v <= 11.2),
            },
        }



    def is_initialized(self) -> bool:
        return self.current_observation is not None

    def _build_scenario_context(self, task_name: str, difficulty: str) -> dict:
        scenarios = {
            ("fault_diagnosis", "easy"): {
                "title": "Routine health scan",
                "objective": "Confirm the vehicle is healthy before dispatch.",
                "grader_focus": "Avoid false positives and identify no-fault situations correctly.",
            },
            ("fault_diagnosis", "medium"): {
                "title": "Low-oil early warning",
                "objective": "Catch maintenance degradation before the vehicle is stranded.",
                "grader_focus": "Low-oil diagnosis should beat generic fault guesses.",
            },
            ("fault_diagnosis", "hard"): {
                "title": "Compound roadside failure",
                "objective": "Prioritize the dominant safety-critical failure in a stacked fault case.",
                "grader_focus": "Overheating must win over secondary low-oil or sensor symptoms.",
            },
            ("driving_decision", "easy"): {
                "title": "Open-road speed management",
                "objective": "Choose a safe immediate maneuver in benign traffic.",
                "grader_focus": "Prefer efficient but safe driving actions.",
            },
            ("driving_decision", "medium"): {
                "title": "Wet-road degradation response",
                "objective": "Balance speed and degraded maintenance health on a slippery road.",
                "grader_focus": "Reward cautious actions without overreacting.",
            },
            ("driving_decision", "hard"): {
                "title": "Emergency degraded braking",
                "objective": "Handle a safety-critical scenario with multiple active failures.",
                "grader_focus": "Immediate risk reduction matters more than efficiency.",
            },
            ("autonomous_control", "easy"): {
                "title": "Steady-state fleet cruise",
                "objective": "Maintain healthy operation while preserving efficiency.",
                "grader_focus": "Reward balanced driving and low-risk operation.",
            },
            ("autonomous_control", "medium"): {
                "title": "Predictive maintenance intervention",
                "objective": "Keep the trip safe while deciding when to slow down or escalate service.",
                "grader_focus": "Reward gradual risk reduction and timely service escalation.",
            },
            ("autonomous_control", "hard"): {
                "title": "Roadside recovery under compound failure",
                "objective": "Recover a degraded vehicle, manage overrides, and coordinate service before collision or breakdown.",
                "grader_focus": "Reward safe stopping, service coordination, and health preservation under pressure.",
            },
        }
        return scenarios[(task_name, difficulty)]

    def _build_initial_telemetry_state(self, difficulty: str) -> TelemetryState:
        vrng = self._difficulty_rng(difficulty)
        latitude, longitude, heading = self._spawn_location(vrng)
        fault_profile = self.vehicle_profile["fault_profile"]

        if difficulty == "easy":
            speed = round(26.0 + vrng.uniform(-6.0, 12.0), 2)
            throttle = round(22.0 + vrng.uniform(-5.0, 10.0), 2)
            engine_temp = round(84.0 + vrng.uniform(-3.0, 4.0), 2)
            oil_level = round(82.0 + vrng.uniform(-7.0, 5.0), 2)
            battery_health = round(88.0 + vrng.uniform(-6.0, 4.0), 2)
            road_condition = "dry"
            failures = FailureState()

            if fault_profile == "battery":
                battery_health = round(56.0 + vrng.uniform(-8.0, 5.0), 2)
                throttle = round(max(10.0, throttle - 4.0), 2)
                failures = FailureState(battery_issue=True)
            elif fault_profile == "oil":
                oil_level = round(48.0 + vrng.uniform(-8.0, 6.0), 2)
                failures = FailureState(low_oil=True)
            elif fault_profile == "safety":
                road_condition = "wet"
            elif fault_profile == "engine":
                engine_temp = round(93.0 + vrng.uniform(-3.0, 5.0), 2)

            base_state = TelemetryState(
                speed=speed,
                rpm=round(1200.0 + vrng.uniform(-120.0, 180.0), 2),
                throttle=throttle,
                gear=self._gear_from_speed(speed),
                engine_load=round(24.0 + vrng.uniform(-4.0, 8.0), 2),
                transmission_load=round(18.0 + vrng.uniform(-3.0, 6.0), 2),
                fuel_rate=round(1.4 + vrng.uniform(-0.2, 0.5), 2),
                acceleration=round(0.6 + vrng.uniform(-0.2, 0.3), 2),
                engine_temp=engine_temp,
                distance_to_obstacle=round((60.0 if fault_profile == "safety" else 85.0) + vrng.uniform(-15.0, 25.0), 2),
                road_condition=road_condition,
                drive_mode=self._drive_mode_from_speed(speed, throttle),
                oil_level=oil_level,
                battery_health=battery_health,
                latitude=latitude,
                longitude=longitude,
                heading=heading,
                failures=failures,
            )
            return self._attach_vehicle_payload(
                telemetry=base_state,
                action_type="continue",
                action_value=0.25,
                fuel_level=round(78.0 + vrng.uniform(-18.0, 8.0), 2),
                odometer_km=round(18432.6 + vrng.uniform(0, 42000), 1),
            )

        if difficulty == "medium":
            speed = round(54.0 + vrng.uniform(-10.0, 14.0), 2)
            throttle = round(34.0 + vrng.uniform(-6.0, 10.0), 2)
            engine_temp = round(95.0 + vrng.uniform(-4.0, 5.0), 2)
            oil_level = round(31.0 + vrng.uniform(-6.0, 4.0), 2)
            battery_health = round(73.0 + vrng.uniform(-8.0, 5.0), 2)
            distance_to_obstacle = round(48.0 + vrng.uniform(-12.0, 20.0), 2)
            road_condition = "wet"
            failures = FailureState(low_oil=True)

            if fault_profile == "battery":
                battery_health = round(34.0 + vrng.uniform(-8.0, 5.0), 2)
                oil_level = round(58.0 + vrng.uniform(-6.0, 6.0), 2)
                engine_temp = round(90.0 + vrng.uniform(-3.0, 4.0), 2)
                failures = FailureState(battery_issue=True)
            elif fault_profile == "engine":
                engine_temp = round(104.0 + vrng.uniform(-3.0, 5.0), 2)
                oil_level = round(52.0 + vrng.uniform(-5.0, 5.0), 2)
                failures = FailureState(engine_overheating=True)
            elif fault_profile == "safety":
                distance_to_obstacle = round(24.0 + vrng.uniform(-8.0, 12.0), 2)
                road_condition = "rain"
                oil_level = round(55.0 + vrng.uniform(-5.0, 5.0), 2)
                failures = FailureState(brake_failure=True, sensor_failure=True)

            base_state = TelemetryState(
                speed=speed,
                rpm=round(2200.0 + vrng.uniform(-200.0, 280.0), 2),
                throttle=throttle,
                gear=self._gear_from_speed(speed),
                engine_load=round(48.0 + vrng.uniform(-6.0, 10.0), 2),
                transmission_load=round(42.0 + vrng.uniform(-5.0, 8.0), 2),
                fuel_rate=round(2.6 + vrng.uniform(-0.3, 0.7), 2),
                acceleration=round(0.3 + vrng.uniform(-0.2, 0.4), 2),
                engine_temp=engine_temp,
                distance_to_obstacle=distance_to_obstacle,
                road_condition=road_condition,
                drive_mode=self._drive_mode_from_speed(speed, throttle),
                oil_level=oil_level,
                battery_health=battery_health,
                latitude=latitude,
                longitude=longitude,
                heading=heading,
                failures=failures,
            )
            return self._attach_vehicle_payload(
                telemetry=base_state,
                action_type="continue",
                action_value=0.35,
                fuel_level=round(46.0 + vrng.uniform(-12.0, 10.0), 2),
                odometer_km=round(51877.2 + vrng.uniform(0, 55000), 1),
            )

        if difficulty == "hard":
            speed = round(68.0 + vrng.uniform(-14.0, 18.0), 2)
            throttle = round(42.0 + vrng.uniform(-8.0, 14.0), 2)
            engine_temp = round(108.0 + vrng.uniform(-2.0, 6.0), 2)
            oil_level = round(24.0 + vrng.uniform(-5.0, 3.0), 2)
            battery_health = round(62.0 + vrng.uniform(-9.0, 4.0), 2)
            distance_to_obstacle = round(68.0 + vrng.uniform(-18.0, 18.0), 2)
            road_condition = "rain"
            failures = FailureState(
                brake_failure=True,
                sensor_failure=True,
                engine_overheating=True,
                low_oil=True,
            )

            if fault_profile == "battery":
                engine_temp = round(94.0 + vrng.uniform(-3.0, 4.0), 2)
                oil_level = round(52.0 + vrng.uniform(-5.0, 5.0), 2)
                battery_health = round(22.0 + vrng.uniform(-6.0, 4.0), 2)
                distance_to_obstacle = round(54.0 + vrng.uniform(-12.0, 14.0), 2)
                failures = FailureState(
                    battery_issue=True,
                    sensor_failure=True,
                )
            elif fault_profile == "oil":
                engine_temp = round(101.0 + vrng.uniform(-2.0, 4.0), 2)
                oil_level = round(14.0 + vrng.uniform(-4.0, 2.0), 2)
                battery_health = round(58.0 + vrng.uniform(-6.0, 4.0), 2)
                failures = FailureState(
                    low_oil=True,
                    brake_failure=True,
                )
            elif fault_profile == "safety":
                engine_temp = round(96.0 + vrng.uniform(-2.0, 4.0), 2)
                oil_level = round(50.0 + vrng.uniform(-6.0, 5.0), 2)
                battery_health = round(60.0 + vrng.uniform(-7.0, 5.0), 2)
                distance_to_obstacle = round(18.0 + vrng.uniform(-8.0, 8.0), 2)
                failures = FailureState(
                    brake_failure=True,
                    sensor_failure=True,
                )

            base_state = TelemetryState(
                speed=speed,
                rpm=round(2580.0 + vrng.uniform(-250.0, 320.0), 2),
                throttle=throttle,
                gear=self._gear_from_speed(speed),
                engine_load=round(58.0 + vrng.uniform(-7.0, 11.0), 2),
                transmission_load=round(46.0 + vrng.uniform(-5.0, 9.0), 2),
                fuel_rate=round(2.8 + vrng.uniform(-0.3, 0.8), 2),
                acceleration=round(0.1 + vrng.uniform(-0.2, 0.5), 2),
                engine_temp=engine_temp,
                distance_to_obstacle=distance_to_obstacle,
                road_condition=road_condition,
                drive_mode=self._drive_mode_from_speed(speed, throttle),
                oil_level=oil_level,
                battery_health=battery_health,
                latitude=latitude,
                longitude=longitude,
                heading=heading,
                failures=failures,
            )
            return self._attach_vehicle_payload(
                telemetry=base_state,
                action_type="continue",
                action_value=0.4,
                fuel_level=round(31.0 + vrng.uniform(-10.0, 8.0), 2),
                odometer_km=round(92341.7 + vrng.uniform(0, 65000), 1),
            )

        raise ValueError("Invalid difficulty")

    def _phase_for_cycle(self) -> str:
        phases = ["engine", "battery", "oil", "safety"]
        start_idx = phases.index(self.vehicle_profile["fault_profile"])
        cycle_idx = (self.episode_state.step_count // 6) % len(phases)
        return phases[(start_idx + cycle_idx) % len(phases)]

    def _apply_fault_rotation(self, telemetry: TelemetryState) -> TelemetryState:
        phase = self._phase_for_cycle()
        self.current_fault_phase = phase

        speed = telemetry.speed
        throttle = telemetry.throttle
        engine_temp = telemetry.engine_temp
        oil_level = telemetry.oil_level
        battery_health = telemetry.battery_health
        distance_to_obstacle = telemetry.distance_to_obstacle
        road_condition = telemetry.road_condition
        failures = telemetry.failures.model_copy()

        if phase == "engine":
            engine_temp = min(150.0, engine_temp + 4.0)
            oil_level = max(18.0, oil_level - 1.5)
            failures = failures.model_copy(update={"engine_overheating": engine_temp >= 104.0})
        elif phase == "battery":
            battery_health = max(8.0, battery_health - 5.5)
            throttle = max(8.0, throttle - 5.0)
            failures = failures.model_copy(update={"battery_issue": battery_health <= 40.0})
        elif phase == "oil":
            oil_level = max(6.0, oil_level - 4.0)
            engine_temp = min(150.0, engine_temp + 1.8)
            failures = failures.model_copy(update={"low_oil": oil_level <= 26.0})
        elif phase == "safety":
            distance_to_obstacle = max(6.0, distance_to_obstacle - 16.0)
            road_condition = "rain" if telemetry.road_condition != "rain" else telemetry.road_condition
            speed = min(160.0, speed + 4.0)
            failures = failures.model_copy(
                update={
                    "brake_failure": distance_to_obstacle < 20.0 or failures.brake_failure,
                    "sensor_failure": True,
                }
            )

        rotated = telemetry.model_copy(
            update={
                "speed": round(speed, 2),
                "throttle": round(throttle, 2),
                "engine_temp": round(engine_temp, 2),
                "oil_level": round(oil_level, 2),
                "battery_health": round(battery_health, 2),
                "distance_to_obstacle": round(distance_to_obstacle, 2),
                "road_condition": road_condition,
                "failures": failures,
                "drive_mode": self._drive_mode_from_speed(speed, throttle),
                "gear": self._gear_from_speed(speed),
            }
        )

        return self._attach_vehicle_payload(
            telemetry=rotated,
            action_type="continue",
            action_value=0.25,
            fuel_level=rotated.vehicle_signals.fuel_level if rotated.vehicle_signals else 50.0,
            odometer_km=rotated.vehicle_signals.odometer_km if rotated.vehicle_signals else 0,
        )

    def _attach_vehicle_payload(
        self,
        telemetry: TelemetryState,
        action_type: str,
        action_value: float,
        fuel_level: float,
        odometer_km: float,
    ) -> TelemetryState:
        signals = build_vehicle_signals(
            speed=telemetry.speed,
            rpm=telemetry.rpm,
            throttle=telemetry.throttle,
            action_type=action_type,
            action_value=action_value,
            gear=telemetry.gear,
            engine_load=telemetry.engine_load,
            transmission_load=telemetry.transmission_load,
            fuel_rate=telemetry.fuel_rate,
            acceleration=telemetry.acceleration,
            engine_temp=telemetry.engine_temp,
            oil_level=telemetry.oil_level,
            battery_health=telemetry.battery_health,
            distance_to_obstacle=telemetry.distance_to_obstacle,
            road_condition=telemetry.road_condition,
            drive_mode=telemetry.drive_mode,
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            heading=telemetry.heading,
            previous_fuel_level=fuel_level,
            previous_odometer_km=odometer_km,
            battery_issue_active=telemetry.failures.battery_issue,
            low_oil_active=telemetry.failures.low_oil,
            dt_seconds=0,
            rng=self.rng,
        )
        events = build_vehicle_events(
            signals=signals,
            failures=telemetry.failures,
            is_collision=False,
        )
        return telemetry.model_copy(
            update={
                "vehicle_signals": signals,
                "vehicle_events": events,
            }
        )

    def _telemetry_to_observation(self, telemetry: TelemetryState) -> Observation:
        return Observation(
            speed=telemetry.speed,
            rpm=telemetry.rpm,
            throttle=telemetry.throttle,
            gear=telemetry.gear,
            engine_load=telemetry.engine_load,
            transmission_load=telemetry.transmission_load,
            fuel_rate=telemetry.fuel_rate,
            acceleration=telemetry.acceleration,
            engine_temp=telemetry.engine_temp,
            distance_to_obstacle=telemetry.distance_to_obstacle,
            road_condition=telemetry.road_condition,
            drive_mode=telemetry.drive_mode,
            oil_level=telemetry.oil_level,
            battery_health=telemetry.battery_health,
            latitude=telemetry.latitude,
            longitude=telemetry.longitude,
            heading=telemetry.heading,
            failures=telemetry.failures,
            history=[],
            vehicle_signals=telemetry.vehicle_signals,
            vehicle_events=telemetry.vehicle_events,
        )

    def reset(self, task_name: str = "fault_diagnosis", difficulty: str = "easy") -> Observation:
        self.current_task = task_name
        self.current_difficulty = difficulty
        self.current_scenario = self._build_scenario_context(task_name, difficulty)

        self.episode_state = EpisodeState(
            step_count=0,
            max_steps=self.max_steps,
            is_collision=False,
            is_engine_failure=False,
            is_safe_stop=False,
        )

        self.current_true_state = self._build_initial_telemetry_state(difficulty=difficulty)
        self.current_observation = self._telemetry_to_observation(self.current_true_state)

        self.override_active = False
        self.override_count = 0
        self.last_action = Action(action_type="continue", value=0.25, reason="background cruise")
        self.last_background_sync_at = time.monotonic()
        self.last_service_booking = None
        self.last_service_recommended = None
        self.pending_service_schedule_request = None
        self.service_autobook_suppressed = False
        self.sequence_id = 0
        self.current_fault_phase = self.vehicle_profile["fault_profile"]
        self.last_done = False
        self.last_reward = strict_task_score(0)
        self.last_reward_breakdown = RewardBreakdown(
            total=strict_task_score(0),
            safety_component=strict_task_score(1),
            efficiency_component=strict_task_score(1),
            diagnosis_component=strict_task_score(0.5),
            service_component=strict_task_score(0),
            health_component=strict_task_score(1),
            sequence_component=strict_task_score(0),
            penalty_component=-1e-2,
        )
        self.last_info = {
            "outcome": "in_progress",
            "collision_risk": 0,
            "override_active": False,
            "override_count": 0,
            "step_count": 0,
            "health_score": 100,
            "alerts": [],
            "active_alerts": [],
            "subsystem_health": {},
            "service_recommended": None,
            "service_booking": None,
            "reward_breakdown": self.last_reward_breakdown.model_dump(),
            "scenario": self.current_scenario,
            "task_score": strict_task_score(0),
            "score": strict_task_score(0),
        }

        self.last_metrics = self._compute_metrics(
            collision_risk=0.02,
            observation=self.current_observation,
        )
        self.last_info = self._build_info(
            observation=self.current_observation,
            collision_risk=0.02,
            action_type="continue",
            action_reason="background cruise",
        )

        return self.current_observation

    def state(self) -> Observation:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")
        self._sync_background_state()
        return self.current_observation

    def _sync_background_state(self) -> None:
        if (
            self.current_observation is None
            or self.current_true_state is None
            or self.current_difficulty is None
        ):
            return

        now = time.monotonic()
        elapsed = now - self.last_background_sync_at
        if elapsed < self.update_interval_seconds:
            return

        steps_to_apply = int(elapsed // self.update_interval_seconds)

        for _ in range(steps_to_apply):
            transition = self.simulator.transition(
                state=self.current_true_state,
                previous_observation=self.current_observation,
                action_type=self.background_action.action_type,
                action_value=self.background_action.value,
                difficulty=self.current_difficulty,
            )

            self.current_true_state = self._apply_fault_rotation(transition["true_state"])
            self.current_observation = self._telemetry_to_observation(self.current_true_state)
            self.last_metrics = self._compute_metrics(
                collision_risk=transition["collision_risk"],
                observation=self.current_observation,
            )
            self.last_reward = self._compute_reward(
                action=self.background_action,
                collision_risk=transition["collision_risk"],
                observation=self.current_observation,
                is_collision=transition["is_collision"],
            )
            self.last_info = self._build_info(
                observation=self.current_observation,
                collision_risk=transition["collision_risk"],
                action_type=self.background_action.action_type,
                action_reason=self.background_action.reason,
            )

        self.last_background_sync_at += steps_to_apply * self.update_interval_seconds

    def compute_health(self, obs: Observation, collision_risk: float) -> int:
        return self._compute_health_snapshot(obs, collision_risk)["overall"]

    def get_alerts(self, obs: Observation, collision_risk: float) -> list[str]:
        return [alert["code"].replace("_", " ") for alert in self._build_active_alerts(obs, collision_risk)]

    def _build_service_payload(
        self,
        observation: Observation,
        health_score: int,
        active_alerts: list[dict],
        action_type: str,
        action_reason: str,
    ) -> tuple[Optional[dict], Optional[dict]]:
        service_recommended = self.last_service_recommended
        service_booking = self.last_service_booking

        severe = (
            health_score < 55
            or any(alert["code"] in {"ENGINE_OVERHEATING", "BRAKE_FAILURE", "BATTERY_ISSUE"} for alert in active_alerts)
        )
        service_due = self._compute_service_due(observation.vehicle_signals.odometer_km)

        if severe or service_due["service_due_now"]:
            service_recommended = find_nearest_service(
                observation.latitude,
                observation.longitude,
            )
            service_recommended = {
                **service_recommended,
                "service_due_km": service_due["next_due_km"],
                "remaining_km": service_due["remaining_km"],
                "vehicle_lat": round(observation.latitude, 6),
                "vehicle_lon": round(observation.longitude, 6),
            }

        if action_type == "reschedule_service":
            parsed_request = self._parse_service_schedule_request(action_reason)
            if parsed_request:
                self.pending_service_schedule_request = parsed_request

        if action_type == "cancel_service":
            service_booking = None
            self.pending_service_schedule_request = None
            self.last_service_booking = None
            self.service_autobook_suppressed = True
            self.last_service_recommended = service_recommended
            return service_recommended, service_booking

        if action_type in {"request_service", "reschedule_service"}:
            self.service_autobook_suppressed = False

        should_book = (
            action_type in {"request_service", "reschedule_service"}
            or (health_score < 35 and not self.service_autobook_suppressed)
        )

        if service_recommended and should_book:
            urgency = "HIGH" if health_score < 35 else "MEDIUM"
            schedule_request = self.pending_service_schedule_request or {}
            schedule = build_service_schedule(
                service_recommended,
                urgency=urgency,
                requested_date=schedule_request.get("requested_date"),
                requested_time=schedule_request.get("requested_time"),
            )
            status = "rescheduled" if action_type == "reschedule_service" and service_booking else "scheduled"
            service_booking = {
                "status": status,
                "booking_id": (
                    service_booking["booking_id"]
                    if service_booking and service_booking.get("booking_id")
                    else f"SRV-{self.vehicle_profile['vin'][-5:]}-{self.sequence_id + 1:03d}"
                ),
                "center_name": service_recommended["name"],
                "center_address": service_recommended.get("address"),
                "center_phone": service_recommended.get("phone"),
                "distance_km": service_recommended["distance_km"],
                "eta_minutes": service_recommended["eta_minutes"],
                "slots_available": service_recommended["slots_available"],
                "urgency": urgency,
                "scheduled_at": schedule["scheduled_at"],
                "scheduled_date": schedule["scheduled_date"],
                "scheduled_time": schedule["scheduled_time"],
                "requested_date": schedule_request.get("requested_date", ""),
                "requested_time": schedule_request.get("requested_time", ""),
                "editable": True,
                "car_id": self.vehicle_id,
                "vehicle_name": self.vehicle_profile["name"],
                "service_center_lat": service_recommended.get("lat"),
                "service_center_lon": service_recommended.get("lon"),
                "vehicle_lat": round(observation.latitude, 6),
                "vehicle_lon": round(observation.longitude, 6),
            }
            self.pending_service_schedule_request = None

        self.last_service_recommended = service_recommended
        self.last_service_booking = service_booking

        return service_recommended, service_booking

    def _build_info(
        self,
        observation: Observation,
        collision_risk: float,
        action_type: str,
        action_reason: str = "",
    ) -> dict:
        health_snapshot = self._compute_health_snapshot(observation, collision_risk)
        health_score = health_snapshot["overall"]
        active_alerts = self._build_active_alerts(observation, collision_risk)
        alerts = [alert["code"].replace("_", " ") for alert in active_alerts]
        service_recommended, service_booking = self._build_service_payload(
            observation=observation,
            health_score=health_score,
            active_alerts=active_alerts,
            action_type=action_type,
            action_reason=action_reason,
        )
        dashboard = self._build_dashboard_payload(
            observation=observation,
            collision_risk=collision_risk,
            health_snapshot=health_snapshot,
            active_alerts=active_alerts,
            service_recommended=service_recommended,
            service_booking=service_booking,
        )
        self.sequence_id += 1

        return {
            "outcome": self.get_episode_outcome(),
            "collision_risk": round(collision_risk, 3),
            "collision_risk_pct": int(round(collision_risk * 100)),
            "override_active": self.override_active,
            "override_count": self.override_count,
            "step_count": self.episode_state.step_count,
            "health_score": health_score,
            "alerts": alerts,
            "active_alerts": active_alerts,
            "subsystem_health": {
                "engine": health_snapshot["engine"],
                "battery": health_snapshot["battery"],
                "safety": health_snapshot["safety"],
                "maintenance": health_snapshot["maintenance"],
            },
            "service_recommended": service_recommended,
            "service_booking": service_booking,
            "reward_breakdown": self.last_reward_breakdown.model_dump(),
            "scenario": self.current_scenario,
            "fault_phase": self.current_fault_phase,
            "car_id": self.vehicle_id,
            "vehicle_name": self.vehicle_profile["name"],
            "dashboard": dashboard,
            "ml_predictions": health_snapshot["ml_predictions"],
            "task_score": self.last_info.get("task_score", strict_task_score(0)),
            "score": self.last_info.get("task_score", strict_task_score(0)),
        }

    def _finish_task_episode(
        self,
        reward: float,
        info_updates: dict,
        metrics: Metrics,
    ) -> StepResult:
        self.last_reward = strict_task_score(reward)
        self.last_done = True
        self.last_metrics = metrics
        self.last_info = {
            **self.last_info,
            **info_updates,
            "override_active": self.override_active,
            "override_count": self.override_count,
            "step_count": self.episode_state.step_count,
            "task_score": strict_task_score(reward),
            "score": strict_task_score(reward),
        }

        return StepResult(
            observation=self.current_observation,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
            metrics=self.last_metrics,
        )

    def _compute_reward(
        self,
        action: Action,
        collision_risk: float,
        observation: Observation,
        is_collision: bool,
    ) -> float:
        if is_collision:
            self.last_reward_breakdown = RewardBreakdown(
                total=strict_task_score(0),
                safety_component=strict_task_score(0),
                efficiency_component=strict_task_score(0),
                diagnosis_component=strict_task_score(0),
                service_component=strict_task_score(0),
                health_component=strict_task_score(0),
                sequence_component=strict_task_score(self.episode_state.step_count / max(1, self.max_steps)),
                penalty_component=-0.99,
            )
            return strict_task_score(0)

        alerts = self._build_active_alerts(observation, collision_risk)
        health_snapshot = self._compute_health_snapshot(observation, collision_risk)
        severe_alert = any(
            alert["code"] in {"ENGINE_OVERHEATING", "LOW_OIL", "BRAKE_FAILURE", "BATTERY_ISSUE"}
            for alert in alerts
        )

        safety = max(1e-2, min(0.99, 1 - collision_risk))
        efficiency = max(0, min(1, observation.speed / 90.0))
        diagnosis = 0.99 if alerts else 0.65
        health_component = max(0, min(1, health_snapshot["overall"] / 100.0))
        sequence = min(1, max(0.2, len(observation.history) / 6.0))

        service_component = 0.55
        if severe_alert and action.action_type in {"request_service", "reschedule_service"}:
            service_component = 0.99
        elif action.action_type == "cancel_service":
            service_component = 0.10 if severe_alert else 0.40
        elif severe_alert and action.action_type in {"brake", "stop"}:
            service_component = 0.78
        elif not severe_alert and action.action_type in {"request_service", "reschedule_service"}:
            service_component = 0.30

        penalty = 0
        if collision_risk > 0.8 and action.action_type == "accelerate":
            penalty -= 0.40
        elif collision_risk > 0.6 and action.action_type == "continue":
            penalty -= 0.18
        if severe_alert and action.action_type == "accelerate":
            penalty -= 0.22

        total = (
            0.28 * safety
            + 0.14 * efficiency
            + 0.12 * diagnosis
            + 0.16 * service_component
            + 0.18 * health_component
            + 0.12 * sequence
            + penalty
        )
        total = strict_task_score(total)

        self.last_reward_breakdown = RewardBreakdown(
            total=strict_task_score(total),
            safety_component=strict_task_score(safety),
            efficiency_component=strict_task_score(efficiency),
            diagnosis_component=strict_task_score(diagnosis),
            service_component=strict_task_score(service_component),
            health_component=strict_task_score(health_component),
            sequence_component=strict_task_score(sequence),
            penalty_component=round(max(-0.99, min(-0.01, penalty)), 3) if penalty < 0 else -0.01,
        )
        return total

    def _compute_metrics(self, collision_risk: float, observation: Observation) -> Metrics:
        safety = 1e-2 if collision_risk >= 0.97 else max(1e-2, min(0.99, 1 - collision_risk))
        efficiency = max(0, min(1, observation.speed / 90.0))

        diagnosis = 0.5
        if (
            observation.engine_temp > 105
            or observation.oil_level < 25
            or observation.battery_health < 30
            or observation.failures.engine_overheating
            or observation.failures.low_oil
            or observation.failures.battery_issue
        ):
            diagnosis = 0.99

        sequence = min(1, self.episode_state.step_count / 10.0)

        return Metrics(
            safety_score=self._strict_metric_score(safety),
            efficiency_score=self._strict_metric_score(efficiency),
            diagnosis_score=self._strict_metric_score(diagnosis),
            sequence_score=self._strict_metric_score(sequence),
        )

    def _check_done(self) -> bool:
        return self.episode_state.check_done()

    def get_episode_outcome(self) -> str:
        if self.episode_state.is_collision:
            return "failure_collision"
        if self.episode_state.is_engine_failure:
            return "failure_engine"
        if self.episode_state.is_safe_stop:
            return "success_safe_stop"
        if self.episode_state.step_count >= self.max_steps:
            return "episode_timeout"
        return "in_progress"

    def step(self, action: Action) -> StepResult:
        if self.current_observation is None or self.current_true_state is None:
            raise RuntimeError("Call reset() before step()")
        if self.current_difficulty is None:
            raise RuntimeError("Call reset() first.")

        if self.current_task == "fault_diagnosis":
            self.last_action = action
            self.episode_state.step_count += 1
            score = grade_fault_diagnosis(action, self.current_observation)
            self.last_reward_breakdown = RewardBreakdown(
                total=strict_task_score(score),
                safety_component=strict_task_score(1),
                efficiency_component=strict_task_score(0),
                diagnosis_component=strict_task_score(score),
                service_component=strict_task_score(0),
                health_component=strict_task_score(self.compute_health(self.current_observation, 0.02) / 100.0),
                sequence_component=strict_task_score(1),
                penalty_component=-1e-2,
            )
            self.last_info = self._build_info(
                observation=self.current_observation,
                collision_risk=0.02,
                action_type=action.action_type,
                action_reason=action.reason,
            )
            return self._finish_task_episode(
                reward=score,
                info_updates={
                    "outcome": "success_diagnosis" if score >= 0.95 else "failure_diagnosis",
                },
                metrics=Metrics(
                    safety_score=self._strict_metric_score(0.99),
                    efficiency_score=self._strict_metric_score(0),
                    diagnosis_score=self._strict_metric_score(score),
                    sequence_score=self._strict_metric_score(1),
                ),
            )

        pre_action_observation = self.current_observation

        self.override_active = self.episode_state.step_count > 0 and self.episode_state.step_count % 3 == 0

        applied_action = action
        if (
            self.override_active
            and action.action_type in ["stop", "brake"]
            and self.current_observation.distance_to_obstacle > 25
        ):
            self.override_count += 1
            applied_action = Action(
                action_type="accelerate",
                value=min(1, max(0.2, action.value)),
                reason="Human override",
            )

        transition = self.simulator.transition(
            state=self.current_true_state,
            previous_observation=self.current_observation,
            action_type=applied_action.action_type,
            action_value=applied_action.value,
            difficulty=self.current_difficulty,
        )

        self.current_true_state = self._apply_fault_rotation(transition["true_state"])
        self.current_observation = self._telemetry_to_observation(self.current_true_state)
        self.last_action = applied_action

        self.episode_state.step_count += 1
        self.episode_state.is_collision |= transition["is_collision"]
        self.episode_state.is_engine_failure |= transition["is_engine_failure"]

        if applied_action.action_type == "stop" and self.current_observation.speed <= 3.0:
            self.episode_state.is_safe_stop = True

        self.last_metrics = self._compute_metrics(
            collision_risk=transition["collision_risk"],
            observation=self.current_observation,
        )
        self.last_reward = self._compute_reward(
            action=applied_action,
            collision_risk=transition["collision_risk"],
            observation=self.current_observation,
            is_collision=transition["is_collision"],
        )
        self.last_done = self._check_done()
        self.last_info = self._build_info(
            observation=self.current_observation,
            collision_risk=transition["collision_risk"],
            action_type=applied_action.action_type,
            action_reason=applied_action.reason,
        )
        self.last_info["task_score"] = strict_task_score(self.last_reward)
        self.last_info["score"] = strict_task_score(self.last_reward)
        self.last_background_sync_at = time.monotonic()

        if self.current_task == "driving_decision":
            decision_score = grade_driving_decision(action, pre_action_observation)
            decision_reward = 0.75 * decision_score + 0.25 * self.last_metrics.safety_score
            self.last_reward_breakdown = RewardBreakdown(
                total=strict_task_score(decision_reward),
                safety_component=strict_task_score(self.last_metrics.safety_score),
                efficiency_component=strict_task_score(self.last_metrics.efficiency_score),
                diagnosis_component=strict_task_score(decision_score),
                service_component=strict_task_score(0),
                health_component=strict_task_score(self.last_info.get("health_score", 0) / 100.0),
                sequence_component=strict_task_score(1),
                penalty_component=-1e-2,
            )
            self.last_info["reward_breakdown"] = self.last_reward_breakdown.model_dump()
            return self._finish_task_episode(
                reward=decision_reward,
                info_updates={
                    "outcome": (
                        "success_safe_decision"
                        if decision_score >= 0.7
                        and not transition["is_collision"]
                        and transition["collision_risk"] < 0.7
                        else "failure_unsafe_decision"
                    ),
                    "decision_score": round(decision_score, 3),
                },
                metrics=Metrics(
                    safety_score=self._strict_metric_score(self.last_metrics.safety_score),
                    efficiency_score=self._strict_metric_score(self.last_metrics.efficiency_score),
                    diagnosis_score=self._strict_metric_score(decision_score),
                    sequence_score=self._strict_metric_score(1),
                ),
            )

        return StepResult(
            observation=self.current_observation,
            reward=self.last_reward,
            done=self.last_done,
            info=self.last_info,
            metrics=self.last_metrics,
        )

    def get_full_state(self) -> dict:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")

        self._sync_background_state()
        active_alerts = self.last_info.get("active_alerts", [])
        dashboard = self.last_info.get("dashboard", {})

        return {
            "vehicle": dashboard.get("vehicle", {
                "car_id": self.vehicle_id,
                "vin": self.vehicle_profile["vin"],
                "name": self.vehicle_profile["name"],
                "maker": self.vehicle_profile["maker"],
            }),
            "observation": self.current_observation.model_dump(),
            "reward": self.last_reward,
            "done": self.last_done,
            "metrics": self.last_metrics.model_dump(),
            "info": self.last_info,
            "dashboard": dashboard,
            "quick_telemetry": dashboard.get("quick_telemetry", {}),
            "trip": dashboard.get("trip", {}),
            "health": dashboard.get("health", {}),
            "ml_predictions": self.last_info.get("ml_predictions", {}),
            "active_alerts": active_alerts,
            "ecu": self._build_ecu_snapshot(
                observation=self.current_observation,
                collision_risk=float(self.last_info.get("collision_risk", 0)),
                active_alerts=active_alerts,
            ),
        }

    def get_ecu_payload(self) -> dict:
        if self.current_observation is None:
            raise RuntimeError("Call reset() first")

        self._sync_background_state()
        return self._build_ecu_snapshot(
            observation=self.current_observation,
            collision_risk=float(self.last_info.get("collision_risk", 0)),
            active_alerts=self.last_info.get("active_alerts", []),
        )
