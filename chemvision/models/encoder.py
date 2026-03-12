"""Dynamic resolution vision encoder with gradient saliency patch selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    """Configuration for :class:`DynamicResolutionEncoder`."""

    vision_model_name_or_path: str = Field(
        "openai/clip-vit-large-patch14-336",
        description="HuggingFace vision encoder ID or local path.",
    )
    grid_size: int = Field(4, gt=0, description="Divide image into grid_size × grid_size patches.")
    base_resolution: int = Field(336, gt=0, description="Default patch resolution in pixels.")
    high_resolution: int = Field(672, gt=0, description="Resolution for high-saliency patches.")
    low_resolution: int = Field(224, gt=0, description="Resolution for low-saliency patches.")
    saliency_top_k: float = Field(
        0.25, ge=0.0, le=1.0, description="Fraction of patches assigned high resolution."
    )
    device: str = Field("cuda", description="Torch device string.")
    dtype: str = Field("bfloat16", description="Model dtype: 'float32', 'float16', or 'bfloat16'.")


@dataclass
class PatchEmbeddings:
    """Container for encoded patch representations.

    Attributes
    ----------
    embeddings:
        ``torch.Tensor`` of shape ``[num_patches, hidden_size]``.
    saliency_scores:
        Scalar saliency for each patch (higher = more informative).
    resolutions:
        Actual pixel resolution used to encode each patch.
    grid_size:
        Side length of the NxN patch grid.
    """

    embeddings: Any  # torch.Tensor [num_patches, hidden_size]
    saliency_scores: list[float]
    resolutions: list[int]
    grid_size: int

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size


class DynamicResolutionEncoder:
    """Vision encoder that allocates resolution budget via gradient saliency.

    The input image is split into a ``grid_size × grid_size`` grid.  A
    lightweight three-layer CNN scorer assigns a scalar saliency to each
    patch using gradient magnitude w.r.t. the scorer input (gradient
    saliency).  The top ``saliency_top_k`` fraction of patches are encoded
    at ``high_resolution``; the rest at ``low_resolution``.  All patches
    are forwarded through the frozen vision backbone and the resulting
    embeddings are stacked and returned.

    Example
    -------
    >>> cfg = EncoderConfig(vision_model_name_or_path="openai/clip-vit-large-patch14-336")
    >>> enc = DynamicResolutionEncoder(cfg)
    >>> enc.load()
    >>> out = enc.encode(pil_image)
    >>> out.embeddings.shape   # [grid_size**2, hidden_size]
    >>> out.saliency_scores    # one float per patch
    """

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self._vision_model: Any = None
        self._processor: Any = None
        self._scorer: Any = None  # nn.Sequential
        self._loaded = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the vision backbone and initialise the lightweight saliency scorer."""
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            ) from exc

        cfg = self.config
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

        self._vision_model = AutoModel.from_pretrained(
            cfg.vision_model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(cfg.device)
        self._vision_model.eval()

        self._processor = AutoProcessor.from_pretrained(cfg.vision_model_name_or_path)

        # Lightweight scorer: 3-layer conv → global avg pool → scalar
        # Kept deliberately small so gradient computation is cheap.
        self._scorer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        ).to(cfg.device)
        self._scorer.eval()

        self._loaded = True

    def encode(self, image: Image) -> PatchEmbeddings:
        """Encode *image* with dynamic per-patch resolution allocation.

        Parameters
        ----------
        image:
            RGB PIL image (any size; will be split into patches internally).

        Returns
        -------
        PatchEmbeddings
            Stacked patch embeddings with saliency and resolution metadata.
        """
        if not self._loaded:
            raise RuntimeError("Call DynamicResolutionEncoder.load() before encode().")

        patches = self._split_patches(image)
        saliency_scores = self._compute_saliency(patches)
        resolutions = self._assign_resolutions(saliency_scores)
        embeddings = self._encode_patches(patches, resolutions)

        return PatchEmbeddings(
            embeddings=embeddings,
            saliency_scores=saliency_scores,
            resolutions=resolutions,
            grid_size=self.config.grid_size,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_patches(self, image: Image) -> list[Image]:
        """Divide *image* into a grid_size × grid_size list of non-overlapping crops."""
        n = self.config.grid_size
        w, h = image.size
        pw, ph = w // n, h // n
        patches: list[Image] = []
        for row in range(n):
            for col in range(n):
                box = (col * pw, row * ph, (col + 1) * pw, (row + 1) * ph)
                patches.append(image.crop(box))
        return patches

    def _compute_saliency(self, patches: list[Image]) -> list[float]:
        """Return gradient-magnitude saliency scores for each patch.

        Each patch is resized to 64×64 and passed through the lightweight
        scorer.  The mean absolute gradient of the scalar output w.r.t. the
        input tensor gives the patch saliency.
        """
        import numpy as np
        import torch

        assert self._scorer is not None, "load() must be called first"

        scores: list[float] = []
        for patch in patches:
            resized = patch.resize((64, 64))
            # PIL → numpy → torch, normalise to [0, 1]
            arr = np.array(resized.convert("RGB")).astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.config.device)
            )
            tensor = tensor.requires_grad_(True)

            scalar = self._scorer(tensor).sum()
            scalar.backward()

            grad = tensor.grad
            assert grad is not None
            scores.append(float(grad.abs().mean()))

        return scores

    def _assign_resolutions(self, saliency_scores: list[float]) -> list[int]:
        """Map each patch index to a pixel resolution based on saliency ranking.

        The top ``saliency_top_k`` fraction receives ``high_resolution``;
        the remainder receives ``low_resolution``.  At least one patch always
        receives the high resolution.
        """
        n = len(saliency_scores)
        top_k = max(1, int(n * self.config.saliency_top_k))
        ranked = sorted(range(n), key=lambda i: saliency_scores[i], reverse=True)
        high_set = set(ranked[:top_k])
        return [
            self.config.high_resolution if i in high_set else self.config.low_resolution
            for i in range(n)
        ]

    def _encode_patches(self, patches: list[Image], resolutions: list[int]) -> Any:
        """Encode each patch through the frozen vision backbone and stack results."""
        import torch

        assert self._vision_model is not None
        assert self._processor is not None

        embeddings: list[Any] = []
        with torch.no_grad():
            for patch, res in zip(patches, resolutions, strict=True):
                resized = patch.resize((res, res))
                inputs = self._processor(images=resized, return_tensors="pt")
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                outputs = self._vision_model(**inputs)

                # Prefer pooled representation; fall back to mean of sequence
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output  # [1, hidden_size]
                else:
                    emb = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
                embeddings.append(emb.squeeze(0))

        return torch.stack(embeddings)  # [num_patches, hidden_size]

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"DynamicResolutionEncoder("
            f"model={self.config.vision_model_name_or_path!r}, "
            f"grid={self.config.grid_size}x{self.config.grid_size}, "
            f"{status})"
        )
