"""Global skill registry — register and retrieve skills by name."""

from __future__ import annotations

from chemvision.skills.base import BaseSkill

_REGISTRY: dict[str, BaseSkill] = {}


def register_skill(skill: BaseSkill) -> BaseSkill:
    """Register a skill instance in the global registry.

    Can be used as a decorator or called directly.

    Example
    -------
    >>> register_skill(SpectrumReadingSkill())
    """
    _REGISTRY[skill.name] = skill
    return skill


def get_skill(name: str) -> BaseSkill:
    """Retrieve a registered skill by name.

    Raises
    ------
    KeyError
        If no skill with ``name`` has been registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"No skill named {name!r}. Available: {list_skills()}")
    return _REGISTRY[name]


def list_skills() -> list[str]:
    """Return all currently registered skill names."""
    return sorted(_REGISTRY.keys())
