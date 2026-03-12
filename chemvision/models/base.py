"""Abstract base class for all vision-language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL.Image import Image

from chemvision.models.config import ModelConfig


class BaseVisionModel(ABC):
    """Common interface every model wrapper must satisfy.

    Subclasses implement :meth:`load` and :meth:`generate` and may
    override :meth:`encode_image` for custom preprocessing.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory."""

    @abstractmethod
    def generate(self, image: Image, prompt: str) -> str:
        """Run a single-turn VQA inference.

        Parameters
        ----------
        image:
            PIL image (already opened by the caller).
        prompt:
            Natural-language instruction or question.

        Returns
        -------
        str
            Model's text response.
        """

    def encode_image(self, image_path: Path) -> Image:
        """Open and pre-validate an image file.

        Override for custom preprocessing (resize, colour-space, etc.).
        """
        from PIL import Image as PILImage

        return PILImage.open(image_path).convert("RGB")

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"{self.__class__.__name__}(model={self.config.model_name_or_path!r}, {status})"
