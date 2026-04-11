import json
import os
from typing import Optional

import requests
from openai import OpenAI

from agent import agent_step
from environment import AutoMindEnv
from models import Action, Metrics, Observation
from tasks import MAX_TASK_SCORE, MIN_TASK_SCORE, evaluate_task, safe_score

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
MAX_STEPS = 20
TEMPERATURE = 0.1
TASK_RUNS = [
    ("fault_diagnosis", "easy"),
    ("fault_diagnosis", "medium"),
    ("fault_diagnosis", "hard"),
    ("driving_decision", "easy"),
    ("driving_decision", "medium"),
    ("driving_decision", "hard"),
    ("autonomous_control", "easy"),
    ("autonomous_control", "medium"),
    ("autonomous_control", "hard"),
]


def strict_score(score: float) -> float:
    return float(round(safe_score(score), 3))


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: float) -> str:
    return f"{strict_score(value):.2f}"


def format_action(action: Action) -> str:
    payload = {
        "action_type": action.action_type,
        "value": round(float(action.value), 3),
        "reason": action.reason,
    }
    return json.dumps(payload, separators=(",", ":"))


def log_start(task: str, env: str, model: str) -> None:
    print("[START]", flush=True)
    print(json.dumps({"task": task, "env": env, "model": model}), flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print("[STEP]", flush=True)
    action_dict = {}
    try:
        action_dict = json.loads(action)
    except json.JSONDecodeError:
        pass
    print(
        json.dumps({
            "step": step,
            "action": action_dict,
            "reward": round(reward, 3),
            "done": done,
            "error": error
        }),
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float], task_name: str) -> None:
    print("[END]", flush=True)
    print(
        json.dumps({
            "task": task_name,
            "success": bool(success),
            "steps": int(steps),
            "score": round(score, 3),
            "rewards": [round(r, 3) for r in rewards]
        }),
        flush=True,
    )


def build_prompt(observation: dict, task_name: str) -> str:
    if task_name == "fault_diagnosis":
        instruction = (
            'Choose exactly one action: "diagnose". Put the predicted fault in "reason". '
            'Valid fault labels: engine_overheating, low_oil, battery_issue, no_fault.'
        )
    elif task_name == "driving_decision":
        instruction = (
            'Choose the single safest next action. Valid actions: brake, accelerate, '
            'turn_left, turn_right, continue, stop.'
        )
    else:
        instruction = (
            "Control the vehicle safely over the episode. Valid actions: brake, accelerate, "
            "turn_left, turn_right, continue, stop, request_service, "
            "reschedule_service, cancel_service."
        )

    return f"""
You are controlling an automotive agent for the task "{task_name}".
Return JSON only in this format:
{{
  "action_type": "string",
  "value": 0.5,
  "reason": "short reason"
}}

{instruction}

Observation:
{json.dumps(observation, indent=2)}
""".strip()


def get_model_action(client: OpenAI, observation: dict, task_name: str) -> Action:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(observation, task_name)}],
        temperature=TEMPERATURE,
        max_tokens=120,
    )
    text = completion.choices[0].message.content or ""
    payload = json.loads(text)
    return Action(
        action_type=str(payload["action_type"]).strip(),
        value=float(payload.get("value", 0.5)),
        reason=str(payload.get("reason", "")).strip(),
    )


class EnvClient:
    def __init__(self) -> None:
        self.local_env: Optional[AutoMindEnv] = None
        self.remote_available = self._check_remote()

    def _check_remote(self) -> bool:
        try:
            response = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            return response.ok
        except Exception:
            return False

    def reset(self, task_name: str, difficulty: str) -> dict:
        if self.remote_available:
            response = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name, "difficulty": difficulty},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["observation"]

        if self.local_env is None:
            self.local_env = AutoMindEnv()
        return self.local_env.reset(task_name=task_name, difficulty=difficulty).model_dump()

    def step(self, action: Action) -> dict:
        if self.remote_available:
            response = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action.model_dump(),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        if self.local_env is None:
            raise RuntimeError("Local environment is not initialized")
        return self.local_env.step(action).model_dump()

    def close(self) -> None:
        self.local_env = None

    def mode(self) -> str:
        return "http" if self.remote_available else "local"


def run_episode(client: EnvClient, llm_client: OpenAI, task_name: str, difficulty: str) -> float:
    rewards: list[float] = []
    steps_taken = 0
    step_idx = 0
    score = MIN_TASK_SCORE
    success = False

    log_start(task=task_name, env=client.mode(), model=MODEL_NAME)

    try:
        obs = client.reset(task_name=task_name, difficulty=difficulty)
        last_action: Optional[Action] = None
        last_metrics: Optional[Metrics] = None
        last_info: Optional[dict] = None
        last_reward = MIN_TASK_SCORE

        for step_idx in range(1, MAX_STEPS + 1):
            observation_obj = Observation(**obs)
            try:
                action = get_model_action(llm_client, obs, task_name=task_name)
            except Exception:
                action = agent_step(observation_obj, task_name=task_name)

            result = client.step(action)
            obs = result["observation"]
            reward = strict_score(float(result["reward"]))
            done = bool(result["done"])
            metrics = result["metrics"]
            info = result["info"]
            error = info.get("last_action_error") if isinstance(info, dict) else None

            rewards.append(reward)
            steps_taken = step_idx
            last_reward = reward
            last_action = action
            last_metrics = Metrics(**metrics)
            last_info = info

            log_step(
                step=step_idx,
                action=format_action(action),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        if last_action is not None and last_metrics is not None:
            score = evaluate_task(
                task_name=task_name,
                action=last_action,
                observation=Observation(**obs),
                metrics=last_metrics,
                info=last_info,
            )
        else:
            score = strict_score(last_reward)

        score = strict_score(score)
        success = score >= 0.7
        return score
    except Exception as exc:
        log_step(
            step=max(step_idx, 1),
            action="{}",
            reward=MIN_TASK_SCORE,
            done=True,
            error=str(exc),
        )
        return MIN_TASK_SCORE
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards, task_name=task_name)


if __name__ == "__main__":
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        client = EnvClient()
        try:
            for task_name, difficulty in TASK_RUNS:
                run_episode(
                    client=client,
                    llm_client=llm_client,
                    task_name=task_name,
                    difficulty=difficulty,
                )
        finally:
            client.close()
    except Exception as exc:
        print(f"[FATAL_ERROR] {exc}")
