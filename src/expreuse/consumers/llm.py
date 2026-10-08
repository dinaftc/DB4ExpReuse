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


class OpenAIChatModel:
    """Any OpenAI-compatible chat endpoint (vLLM, Ollama, hosted APIs)."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 64):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens)
        return response.choices[0].message.content or ""


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
