"""LLaVA-1.6 and InternVL2 wrapper using HuggingFace transformers AutoModel."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.models.base import BaseVisionModel
from chemvision.models.config import ModelConfig


class LLaVAWrapper(BaseVisionModel):
    """HuggingFace wrapper for LLaVA-1.6 and InternVL2 family models.

    Dispatches to ``LlavaNextForConditionalGeneration`` for LLaVA models
    and to ``AutoModelForCausalLM`` (with ``trust_remote_code=True``) for
    InternVL2, selecting automatically from the model identifier string.

    Example
    -------
    >>> cfg = ModelConfig(model_name_or_path="llava-hf/llava-v1.6-mistral-7b-hf")
    >>> model = LLaVAWrapper(cfg)
    >>> model.load()
    >>> answer = model.generate(image, "Identify the functional groups.")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # BaseVisionModel interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights and processor from the HuggingFace hub."""
        try:
            import torch
            from transformers import AutoProcessor
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
        attn_impl = "flash_attention_2" if cfg.use_flash_attention else "eager"

        if self._is_internvl(cfg.model_name_or_path):
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            ).to(cfg.device)
        else:
            from transformers import LlavaNextForConditionalGeneration

            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                cfg.model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            ).to(cfg.device)

        self._processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=True
        )
        self._model.eval()
        self._loaded = True

    def generate(self, image: Image, prompt: str) -> str:
        """Run a single-turn VQA inference and return the decoded text response."""
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError("Call load() before generate().")

        import torch

        messages = self._build_messages(image, prompt)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0.0,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0, input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, image: Image, prompt: str) -> list[dict]:
        """Construct the chat-template message list for a single user turn."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    @staticmethod
    def _is_internvl(model_id: str) -> bool:
        """Return ``True`` if *model_id* identifies an InternVL2 model."""
        lower = model_id.lower()
        return "internvl" in lower or "intern_vl" in lower or "internlm" in lower
