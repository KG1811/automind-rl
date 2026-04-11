from environment import AutoMindEnv
from models import Action


def test_reward_breakdown_and_scenario_present():
    env = AutoMindEnv(seed=12, vehicle_id="readiness-car")
    env.reset("autonomous_control", "medium")
    result = env.step(Action(action_type="brake", value=0.6, reason="stabilize"))

    assert "reward_breakdown" in result.info
    assert "scenario" in result.info
    assert result.info["scenario"]["title"]
    assert -1 <= result.info["reward_breakdown"]["total"] <= 1


def test_tasks_and_difficulties_reset_cleanly():
    tasks = ["fault_diagnosis", "driving_decision", "autonomous_control"]
    difficulties = ["easy", "medium", "hard"]

    for task_name in tasks:
        for difficulty in difficulties:
            env = AutoMindEnv(seed=21, vehicle_id=f"{task_name}-{difficulty}")
            observation = env.reset(task_name, difficulty)

            assert observation.drive_mode in {"idle", "city", "cruise", "sport"}
            assert 0 <= observation.engine_temp <= 150.0
            assert 0 <= observation.vehicle_signals.battery_voltage <= 18.0
