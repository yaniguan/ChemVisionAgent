"""Tests for chemvision.skills module."""

import pytest

from chemvision.skills.registry import _REGISTRY, get_skill, list_skills, register_skill
from chemvision.skills.base import BaseSkill, SkillResult
from typing import Any
from PIL.Image import Image


class _DummySkill(BaseSkill):
    name = "dummy"

    def build_prompt(self, **kwargs: Any) -> str:
        return "dummy prompt"

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> SkillResult:
        return SkillResult(skill_name=self.name, raw_output="ok")


def test_register_and_retrieve() -> None:
    skill = _DummySkill()
    register_skill(skill)
    assert get_skill("dummy") is skill


def test_list_skills_includes_registered() -> None:
    register_skill(_DummySkill())
    assert "dummy" in list_skills()


def test_get_skill_unknown_raises() -> None:
    with pytest.raises(KeyError, match="no_such_skill"):
        get_skill("no_such_skill")


def test_skill_result_schema() -> None:
    result = SkillResult(skill_name="test", raw_output="hello", confidence=0.95)
    assert result.confidence == pytest.approx(0.95)
