"""Operator libraries and signatures (Table 2).

Operators are stored under a canonical name so that applications naming the
same operation differently (take / pick up, put / move) share subtraces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SIGNATURES: dict[str, tuple[str, ...]] = {
    "take": ("object", "source"),
    "put": ("object", "destination"),
    "heat": ("object", "appliance"),
    "cool": ("object", "appliance"),
    "clean": ("object", "appliance"),
    "use": ("appliance",),
    "measure": ("object", "instrument"),
    "pour": ("object", "destination"),
    "activate": ("device",),
    "deactivate": ("device",),
    "connect": ("component_1", "component_2"),
    "mix": ("container",),
    "focus on": ("object",),
    "read": ("object",),
}

ALIASES: dict[str, str] = {"pick up": "take", "move": "put"}


def canonical(operator: str) -> str:
    name = " ".join(operator.lower().split())
    return ALIASES.get(name, name)


def signature(operator: str) -> tuple[str, ...]:
    return SIGNATURES[canonical(operator)]


@dataclass(frozen=True)
class OperatorLibrary:
    """The finite operator vocabulary of one application."""

    application: str
    operators: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        ops = frozenset(canonical(o) for o in self.operators)
        unknown = ops - SIGNATURES.keys()
        if unknown:
            raise ValueError(f"operators without a signature: {sorted(unknown)}")
        object.__setattr__(self, "operators", ops)

    def __contains__(self, operator: str) -> bool:
        return canonical(operator) in self.operators

    def shared(self, other: "OperatorLibrary") -> frozenset[str]:
        return self.operators & other.operators


ALFWORLD = OperatorLibrary("alfworld", frozenset(
    {"take", "put", "heat", "cool", "clean", "use"}))
SCIENCEWORLD = OperatorLibrary("scienceworld", frozenset(
    {"pick up", "move", "heat", "cool", "measure", "pour", "activate",
     "deactivate", "connect", "mix", "focus on", "read"}))
