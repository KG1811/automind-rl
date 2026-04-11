"""
AutoMind OpenEnv — Baseline Inference Script
Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Uses OpenAI client for all LLM calls.
Emits [START] / [STEP] / [END] stdout logs per Meta spec.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import requests
from openai import OpenAI

from agent import agent_step
from environment import AutoMindEnv
from models import Action, Metrics, Observation
from tasks import evaluate_task

# ── Env vars ───────────────────────────────────────────────────────────────────
API_BASE_URL: str        = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str          = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str]  = os.getenv("HF_TOKEN")
ENV_BASE_URL: str        = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_STEPS:   int   = 20
TEMPERATURE: float = 0.1
BENCHMARK:   str   = "automind-rl"

TASK_RUNS = [
    ("fault_diagnosis",    "easy"),
    ("fault_diagnosis",    "medium"),
    ("fault_diagnosis",    "hard"),
    ("driving_decision",   "easy"),
    ("driving_decision",   "medium"),
    ("driving_decision",   "hard"),
    ("autonomous_control", "easy"),
    ("autonomous_control", "medium"),
    ("autonomous_control", "hard"),
]


# ── Score safety ───────────────────────────────────────────────────────────────

def safe_score(x: float) -> float:
    """Guarantee score strictly within (0, 1) — never 0.0 or 1.0."""
    return max(0.05, min(0.95, float(x)))


# ── Log helpers — EXACT Meta format ───────────────────────────────────────────

def _fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def _fmt_reward(v: float) -> str:
    return f"{safe_score(v):.2f}"


def _fmt_action(action: Action) -> str:
    return json.dumps(
        {
            "action_type": action.action_type,
            "value": round(float(action.value), 3),
            "reason": action.reason,
        },
        separators=(",", ":"),
    )


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = "null" if error is None else str(error).replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} reward={_fmt_reward(reward)} "
        f"done={_fmt_bool(done)} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """
    Meta spec [END] line — MUST include score= field.
    Format: [END] success=... steps=... score=... rewards=...
    """
    reward_str = ",".join(_fmt_reward(r) for r in rewards)
    print(
        f"[END] success={_fmt_bool(success)} steps={steps} "
        f"score={_fmt_reward(score)} rewards={reward_str}",
        flush=True,
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

def build_prompt(observation: dict, task_name: str) -> str:
    if task_name == "fault_diagnosis":
        instruction = (
            'Use action_type "diagnose". Set "reason" to the predicted fault.\n'
            "Valid: engine_overheating | low_oil | battery_issue | no_fault"
        )
    elif task_name == "driving_decision":
        instruction = (
            "Pick the single safest action.\n"
            "Valid: brake | accelerate | turn_left | turn_right | continue | stop"
        )
    else:
        instruction = (
            "Control vehicle safely over the episode.\n"
            "Valid: brake | accelerate | turn_left | turn_right | continue | stop "
            "| request_service | reschedule_service | cancel_service"
        )

    return (
        f'You are an automotive agent for task "{task_name}".\n'
        f"Reply with JSON ONLY — no markdown, no explanation:\n"
        f'{{"action_type":"string","value":0.5,"reason":"short reason"}}\n\n'
        f"{instruction}\n\n"
        f"Observation:\n{json.dumps(observation, indent=2)}"
    )


# ── LLM call ───────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    obs_dict: dict,
    task_name: str,
    obs_obj: Observation,
) -> Action:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs_dict, task_name)}],
            temperature=TEMPERATURE,
            max_tokens=150,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps response
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        payload = json.loads(raw.strip())
        return Action(
            action_type=str(payload.get("action_type", "continue")).strip(),
            value=safe_score(float(payload.get("value", 0.5))),
            reason=str(payload.get("reason", "")).strip(),
        )
    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", flush=True)
        return agent_step(obs_obj, task_name=task_name)


# ── Env HTTP client ────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self) -> None:
        self.local_env: Optional[AutoMindEnv] = None
        self.remote_available = self._check_remote()

    def _check_remote(self) -> bool:
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            return r.ok
        except Exception:
            return False

    def reset(self, task_name: str, difficulty: str) -> dict:
        if self.remote_available:
            r = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name, "difficulty": difficulty},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["observation"]
        if self.local_env is None:
            self.local_env = AutoMindEnv()
        return self.local_env.reset(
            task_name=task_name, difficulty=difficulty
        ).model_dump()

    def step(self, action: Action) -> dict:
        if self.remote_available:
            r = requests.post(
                f"{ENV_BASE_URL}/step",
                json=action.model_dump(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        if self.local_env is None:
            raise RuntimeError("Local env not initialized")
        return self.local_env.step(action).model_dump()

    def close(self) -> None:
        self.local_env = None

    def mode(self) -> str:
        return "http" if self.remote_available else "local"


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    env_client: EnvClient,
    llm_client: OpenAI,
    task_name: str,
    difficulty: str,
) -> float:
    rewards:     list[float]       = []
    steps_taken: int               = 0
    step_idx:    int               = 0
    score:       float             = 0.05
    success:     bool              = False

    log_start(task=task_name, env=env_client.mode(), model=MODEL_NAME)

    try:
        obs = env_client.reset(task_name=task_name, difficulty=difficulty)

        last_action:  Optional[Action]  = None
        last_metrics: Optional[Metrics] = None
        last_info:    Optional[dict]    = None
        last_reward:  float             = 0.05

        for step_idx in range(1, MAX_STEPS + 1):
            obs_obj = Observation(**obs)
            action  = get_model_action(llm_client, obs, task_name, obs_obj)

            result  = env_client.step(action)
            obs     = result["observation"]
            reward  = safe_score(float(result["reward"]))
            done    = bool(result["done"])
            metrics_raw = result.get("metrics") or {}
            info        = result.get("info") or {}
            error       = info.get("last_action_error") if isinstance(info, dict) else None

            rewards.append(reward)
            steps_taken  = step_idx
            last_reward  = reward
            last_action  = action
            last_metrics = Metrics(**metrics_raw) if metrics_raw else None
            last_info    = info

            log_step(
                step=step_idx,
                action=_fmt_action(action),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Final score via grader
        if last_action is not None and last_metrics is not None:
            score = safe_score(
                evaluate_task(
                    task_name=task_name,
                    action=last_action,
                    observation=Observation(**obs),
                    metrics=last_metrics,
                    info=last_info,
                )
            )
        else:
            score = safe_score(last_reward)

        success = score >= 0.70

    except Exception as exc:
        log_step(
            step=max(step_idx, 1),
            action="{}",
            reward=0.05,
            done=True,
            error=str(exc),
        )
        score = 0.05

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env_client = EnvClient()

    try:
        for task_name, difficulty in TASK_RUNS:
            run_episode(
                env_client=env_client,
                llm_client=llm_client,
                task_name=task_name,
                difficulty=difficulty,
            )
    finally:
        env_client.close()