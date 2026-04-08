from models import Action
from environment import AutoMindEnv
from main import get_env


def test_dashboard_payload_exposes_realistic_summary():
    env = AutoMindEnv(seed=7, vehicle_id="car-alpha")
    env.reset("autonomous_control", "hard")

    state = env.get_full_state()
    dashboard = state["dashboard"]

    assert dashboard["quick_telemetry"]["engine_temp_c"] <= 150.0
    assert dashboard["trip"]["range_km"] >= 0
    assert dashboard["trip"]["drive_mode_display"] in {"Parked", "City", "Cruise", "Sport"}
    assert dashboard["health"]["overall_score"] < 80
    assert dashboard["health"]["engine_status"] != "NORMAL"
    assert len(state["info"]["active_alerts"]) >= 1


def test_service_booking_persists_after_request():
    env = AutoMindEnv(seed=9, vehicle_id="service-car")
    env.reset("autonomous_control", "hard")

    result = env.step(Action(action_type="request_service", value=1.0, reason="book service"))
    assert result.info["service_booking"]["status"] == "scheduled"

    state = env.get_full_state()
    assert state["info"]["service_booking"]["status"] == "scheduled"


def test_different_car_ids_have_distinct_vehicle_profiles_and_state():
    env_one = get_env("car-one")
    env_two = get_env("car-two")

    state_one = env_one.get_full_state()
    state_two = env_two.get_full_state()

    assert state_one["vehicle"]["vin"] != state_two["vehicle"]["vin"]
    assert (
        state_one["observation"]["vehicle_signals"]["odometer_km"]
        != state_two["observation"]["vehicle_signals"]["odometer_km"]
    )
