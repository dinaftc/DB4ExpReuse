"""Grounding (Section 3.3): map the entities of a stored binding to the
entities bound to the same operator roles in the target task, and substitute
them into the stored actions."""
from __future__ import annotations

import re
from dataclasses import replace

from .model import Experience, Plan, Step


def entity_mapping(stored: dict[str, str], target: dict[str, str]) -> dict[str, str]:
    """Stored entity -> target entity, for the roles bound on both sides."""
    return {stored[role]: target[role] for role in stored
            if role in target and stored[role] != target[role]}


def substitute(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-entity occurrences in one pass, so that a mapping such as
    {mug 2 -> mug 4, mug 4 -> mug 1} does not chain."""
    if not mapping:
        return text
    names = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, names)) + r")(?![\w])")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def _apply(steps: list[Step], mapping: dict[str, str], observations: bool) -> list[Step]:
    return [replace(s, action=substitute(s.action, mapping),
                    observation=substitute(s.observation, mapping) if observations
                    else s.observation) for s in steps]


def ground(experience: Experience, plan: Plan, observations: bool = False) -> Experience:
    """Ground a query answer against the target plan's bindings.

    A subtrace hit at plan position i is grounded against the binding of
    position i. A complete trace is grounded by aligning its subtraces with the
    plan positions of the same operator, in order; a stored entity keeps the
    first target entity it is mapped to.
    """
    if experience.route == "subtraces":
        for position, hits in enumerate(experience.subtraces):
            target = plan[position].binding
            for hit in hits:
                mapping = entity_mapping(hit.binding, target)
                hit.steps = _apply(hit.steps, mapping, observations)
                hit.binding = {r: mapping.get(e, e) for r, e in hit.binding.items()}
    elif experience.route == "whole":
        for trace in experience.traces:
            mapping: dict[str, str] = {}
            free = list(range(len(plan)))
            for sub in trace.subtraces:
                position = next((i for i in free if plan[i].operator == sub.operator), None)
                if position is None:
                    continue
                free.remove(position)
                for src, dst in entity_mapping(sub.binding, plan[position].binding).items():
                    mapping.setdefault(src, dst)
            trace.steps = _apply(trace.steps, mapping, observations)
    return experience
