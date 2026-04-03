"""Abstract base class for all composable vision skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel, Field


class SkillResult(BaseModel):
    """Generic skill output container."""

    skill_name: str
    raw_output: str
    parsed: dict[str, Any] = {}
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class BaseSkill(ABC):
    """A composable, stateless vision capability unit.

    Skills are pure functions wrapped in a class to allow configuration
    and registration. They do NOT hold model state — the model is passed
    at call time so skills can be mixed and matched across model backends.

    Example
    -------
    >>> skill = MySkill(name="my_skill")
    >>> result = skill(image, model)
    """

    name: str  # must be set by subclass

    @abstractmethod
    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SkillResult:
        """Execute the skill on a single image.

        Parameters
        ----------
        image:
            PIL image to analyse.
        model:
            A loaded :class:`~chemvision.models.base.BaseVisionModel` instance.
        **kwargs:
            Skill-specific keyword arguments (e.g. ``prompt_override``).
        """

    def build_prompt(self, **kwargs: Any) -> str:
        """Return the default prompt for this skill.

        Override to customise the instruction sent to the model.
        """
        raise NotImplementedError
