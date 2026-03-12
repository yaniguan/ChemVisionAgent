"""Anthropic vision fallback: duck-typed vision model using the Anthropic API.

When no local HuggingFace model is configured, skills can use this adapter
to call ``claude-sonnet-4-20250514`` (or any vision-capable Claude model)
as a drop-in replacement for a local ``BaseVisionModel``.

The class exposes a single ``generate(image, prompt) -> str`` method that
matches the interface expected by all :class:`~chemvision.skills.base.BaseSkill`
implementations.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL.Image import Image


class AnthropicVisionFallback:
    """Duck-typed vision model backed by the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    model:
        Claude model to use (must support vision).
    max_tokens:
        Maximum number of tokens to generate.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy client initialisation
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "anthropic package is required for AnthropicVisionFallback. "
                    "Install it with: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API (matches BaseVisionModel.generate signature)
    # ------------------------------------------------------------------

    def generate(self, image: Image, prompt: str) -> str:
        """Call the Anthropic vision API and return the model's text response.

        Parameters
        ----------
        image:
            PIL image to analyse.
        prompt:
            Instruction / question to send alongside the image.

        Returns
        -------
        str
            Raw text content of the first response block.
        """
        client = self._get_client()

        # Encode image as base64 PNG
        buf = io.BytesIO()
        rgb_image = image.convert("RGB")
        rgb_image.save(buf, format="PNG")
        b64_data = base64.standard_b64encode(buf.getvalue()).decode()

        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        # Extract plain text from the response
        text_blocks = [
            block.text
            for block in message.content
            if hasattr(block, "text")
        ]
        return "\n".join(text_blocks)
