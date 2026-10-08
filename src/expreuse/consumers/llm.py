"""Language-model consumer: the returned experience is given as context."""
from __future__ import annotations

import re
from typing import Protocol, Sequence

from ..model import Experience, Plan, Task
from .base import History

CROSS_APPLICATION_NOTE = (
    "The experience below was recorded in a different application. Its actions "
    "are shown as recorded and may not be valid here: adapt them to this "
    "application instead of copying them.")


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def _usage_counter() -> dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


class OpenAIChatModel:
    """Any OpenAI-compatible chat endpoint (vLLM, Ollama, hosted APIs). `extra` goes into the request body
    (for OpenRouter: the provider order, the reasoning switch, see `openrouter_options`). `usage` accumulates
    the tokens of all calls."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 64, timeout: float | None = None,
                 retries: int = 0, extra: dict | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=retries)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.extra = extra or {}
        self.usage = _usage_counter()

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens, **({"extra_body": self.extra} if self.extra else {}))
        usage = getattr(response, "usage", None)
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return response.choices[0].message.content or ""


def openrouter_options(providers: Sequence[str] = (), reasoning: bool = True) -> dict:
    """OpenRouter's request options: serve the model from the listed providers in that order, and switch off
    its reasoning mode when `reasoning` is false."""
    options: dict = {}
    if providers:
        options["provider"] = {"order": list(providers), "allow_fallbacks": False}
    if not reasoning:
        options["reasoning"] = {"enabled": False}
    return options


class BedrockChatModel:
    """Amazon Bedrock's Converse API (boto3); region and credentials come from the AWS environment."""

    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 64,
                 timeout: float | None = None, retries: int = 0, region: str | None = None):
        import boto3
        from botocore.config import Config

        config = Config(read_timeout=timeout, retries={"max_attempts": retries + 1, "mode": "standard"})
        self.client = boto3.client("bedrock-runtime", config=config, region_name=region)
        self.model_id, self.temperature, self.max_tokens = model_id, temperature, max_tokens
        self.usage = _usage_counter()

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        turns = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        response = self.client.converse(
            modelId=self.model_id, messages=turns, system=system,
            inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens})
        usage = response.get("usage", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += usage.get("inputTokens", 0)
        self.usage["completion_tokens"] += usage.get("outputTokens", 0)
        return "".join(part.get("text", "") for part in response["output"]["message"]["content"])


def _render_steps(steps) -> str:
    return "\n".join(f"> {s.action}\n{s.observation}" for s in steps)


def render_experience(experience: Experience) -> str:
    """Text form of a query answer, in plan order for subtraces."""
    if experience.route == "whole":
        return "\n\n".join(
            f"Past task: {t.goal}\n{_render_steps(t.steps)}" for t in experience.traces)
    blocks = []
    for hits in experience.subtraces:
        for h in hits:
            blocks.append(f"How to {h.operator} (from past task: {h.goal})\n"
                          f"{_render_steps(h.steps)}")
    return "\n\n".join(blocks)


class LLMConsumer:
    def __init__(self, model: ChatModel, system_prompt: str, with_plan: bool = False,
                 list_admissible: bool = False):
        self.model = model
        self.system_prompt = system_prompt
        self.with_plan = with_plan
        self.list_admissible = list_admissible
        self._context = ""

    def begin(self, task: Task, plan: Plan, experience: Experience,
              cross_application: bool = False) -> None:
        parts = [f"Task: {task.goal}"]
        if self.with_plan and plan:
            parts.append("Plan: " + ", ".join(p.operator for p in plan))
        if experience:
            if cross_application:
                parts.append(CROSS_APPLICATION_NOTE)
            parts.append("Relevant past experience:\n" + render_experience(experience))
        self._context = "\n\n".join(parts)

    def messages(self, history: History, admissible: Sequence[str]) -> list[dict[str, str]]:
        lines = [history[0][1]] if history else []
        lines += [f"> {a}\n{o}" for a, o in history[1:]]
        prompt = self._context + "\n\nCurrent episode:\n" + "\n".join(lines)
        if self.list_admissible and admissible:
            prompt += "\n\nValid actions:\n" + "\n".join(admissible)
        prompt += "\n\nReply with the next action only."
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}]

    def act(self, history: History, admissible: Sequence[str]) -> str:
        reply = self.model.complete(self.messages(history, admissible)).strip()
        match = re.search(r"ACTION:\s*(.+)", reply)
        line = match.group(1) if match else (reply.splitlines()[0] if reply else "")
        return line.strip().lstrip("> ").strip().strip("'\"")
