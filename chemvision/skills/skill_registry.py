"""SkillRegistry: OOP wrapper around the global skill registry.

The class shares its backing store with the module-level ``_REGISTRY`` dict in
:mod:`chemvision.skills.registry`, so both the functional API
(``get_skill``, ``list_skills``) and the object-oriented API
(:class:`SkillRegistry`) stay in sync.

A singleton :data:`DEFAULT_REGISTRY` is created at module import time with all
five built-in skills pre-registered.
"""

from __future__ import annotations

from typing import Iterator

from chemvision.skills.base import BaseSkill
from chemvision.skills.registry import _REGISTRY, get_skill, list_skills, register_skill


class SkillRegistry:
    """Object-oriented interface to the global skill registry.

    :class:`SkillRegistry` wraps the same ``_REGISTRY`` dict used by the
    module-level helpers, so adding a skill through this class also makes it
    accessible via :func:`~chemvision.skills.registry.get_skill`.

    Example
    -------
    >>> registry = SkillRegistry()
    >>> registry.register(MyCustomSkill())
    >>> registry["my_custom_skill"]
    <MyCustomSkill>
    >>> "analyze_structure" in registry
    True
    >>> registry.list_skills()
    ['analyze_structure', 'compare_structures', ...]
    """

    def __init__(self) -> None:
        # Share the module-level dict as backing store
        self._store: dict[str, BaseSkill] = _REGISTRY

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, skill: BaseSkill) -> BaseSkill:
        """Register *skill* and return it (usable as a decorator).

        Parameters
        ----------
        skill:
            Any :class:`~chemvision.skills.base.BaseSkill` instance.

        Returns
        -------
        BaseSkill
            The same *skill* that was passed in.
        """
        return register_skill(skill)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseSkill:
        """Return the skill registered under *name*.

        Raises
        ------
        KeyError
            When no skill with *name* has been registered.
        """
        return get_skill(name)

    def list_skills(self) -> list[str]:
        """Return a sorted list of all registered skill names."""
        return list_skills()

    # ------------------------------------------------------------------
    # Pythonic helpers
    # ------------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._store.keys()))

    def __getitem__(self, name: str) -> BaseSkill:
        return self.get(name)

    def __repr__(self) -> str:
        names = ", ".join(self.list_skills())
        return f"SkillRegistry([{names}])"


# ---------------------------------------------------------------------------
# Auto-register the five built-in skills
# ---------------------------------------------------------------------------
# Imported here (not at package level) to avoid circular-import issues.

from chemvision.skills.analyze_structure import AnalyzeStructureSkill  # noqa: E402
from chemvision.skills.compare_structures import CompareStructuresSkill  # noqa: E402
from chemvision.skills.detect_anomaly import DetectAnomalySkill  # noqa: E402
from chemvision.skills.extract_spectrum import ExtractSpectrumSkill  # noqa: E402
from chemvision.skills.validate_caption import ValidateCaptionSkill  # noqa: E402

DEFAULT_REGISTRY: SkillRegistry = SkillRegistry()
for _skill in [
    AnalyzeStructureSkill(),
    ExtractSpectrumSkill(),
    CompareStructuresSkill(),
    ValidateCaptionSkill(),
    DetectAnomalySkill(),
]:
    DEFAULT_REGISTRY.register(_skill)
