"""Vision model wrappers and fine-tuning utilities.

Public API
----------
BaseVisionModel           -- abstract base for all model backends
LLaVAWrapper              -- wrapper for LLaVA-1.6 and InternVL2 models
QwenVLWrapper             -- wrapper for Qwen-VL family models
DynamicResolutionEncoder  -- resolution-adaptive patch encoder with gradient saliency
ChainOfVisionReasoning    -- 3-step structured reasoning wrapper
PeftFineTuner             -- LoRA / QLoRA fine-tuning orchestrator
ModelConfig               -- base model configuration
EncoderConfig             -- encoder configuration
PeftConfig                -- LoRA fine-tuning configuration
"""

from chemvision.models.base import BaseVisionModel
from chemvision.models.config import ModelConfig, PeftConfig
from chemvision.models.encoder import DynamicResolutionEncoder, EncoderConfig, PatchEmbeddings
from chemvision.models.finetuner import PeftFineTuner
from chemvision.models.llava import LLaVAWrapper
from chemvision.models.qwen_vl import QwenVLWrapper
from chemvision.models.reasoning import (
    AnalysisResult,
    BoundingBox,
    ChainOfVisionOutput,
    ChainOfVisionReasoning,
    ConclusionResult,
    LocalizationResult,
)

__all__ = [
    "BaseVisionModel",
    "ModelConfig",
    "PeftConfig",
    "EncoderConfig",
    "PatchEmbeddings",
    "DynamicResolutionEncoder",
    "ChainOfVisionReasoning",
    "ChainOfVisionOutput",
    "BoundingBox",
    "LocalizationResult",
    "AnalysisResult",
    "ConclusionResult",
    "PeftFineTuner",
    "LLaVAWrapper",
    "QwenVLWrapper",
]
