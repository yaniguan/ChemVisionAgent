"""Wrapper for the Qwen-VL family of vision-language models."""

from __future__ import annotations

from PIL.Image import Image

from chemvision.models.base import BaseVisionModel
from chemvision.models.config import ModelConfig


class QwenVLWrapper(BaseVisionModel):
    """HuggingFace-based wrapper for Qwen2-VL / Qwen-VL-Chat models.

    Example
    -------
    >>> cfg = ModelConfig(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    >>> model = QwenVLWrapper(cfg)
    >>> model.load()
    >>> answer = model.generate(image, "What functional groups are present?")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load Qwen-VL weights and processor from HuggingFace hub."""
        raise NotImplementedError

    def generate(self, image: Image, prompt: str) -> str:
        """Run Qwen-VL inference and return the decoded text response."""
        raise NotImplementedError

    def _build_messages(self, image: Image, prompt: str) -> list[dict]:
        """Construct the chat-template message list expected by Qwen-VL."""
        raise NotImplementedError
