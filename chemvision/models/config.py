"""Model configuration schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for loading a vision-language model."""

    model_name_or_path: str = Field(..., description="HuggingFace model ID or local path.")
    device: str = Field("cuda", description="Torch device string, e.g. 'cuda' or 'cpu'.")
    dtype: str = Field("bfloat16", description="Model dtype: 'float32', 'float16', or 'bfloat16'.")
    max_new_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    use_flash_attention: bool = Field(True)


class PeftConfig(BaseModel):
    """Configuration for PEFT / LoRA fine-tuning."""

    base_model: ModelConfig
    lora_r: int = Field(16, description="LoRA rank.")
    lora_alpha: int = Field(32)
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    output_dir: str = Field("checkpoints/")
    num_train_epochs: int = Field(3, gt=0)
    per_device_train_batch_size: int = Field(2, gt=0)
    gradient_accumulation_steps: int = Field(8, gt=0)
    learning_rate: float = Field(2e-4, gt=0.0)
