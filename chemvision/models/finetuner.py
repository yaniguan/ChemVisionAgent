"""LoRA / QLoRA fine-tuning orchestrator using PEFT + HuggingFace Trainer."""

from __future__ import annotations

from typing import Any

from chemvision.data.schema import ImageRecord
from chemvision.models.config import PeftConfig


class PeftFineTuner:
    """Orchestrate LoRA fine-tuning on a ChemVision dataset split.

    Uses HuggingFace ``Trainer`` with ``peft`` LoRA adapters applied to the
    base VLM.  Training metrics are reported to Weights & Biases when the
    ``wandb`` package is available.

    Example
    -------
    >>> cfg = PeftConfig(base_model=ModelConfig(...), output_dir="runs/lora-v1")
    >>> tuner = PeftFineTuner(cfg)
    >>> tuner.train(train_records, val_records)
    """

    def __init__(self, config: PeftConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_model(self) -> Any:
        """Load the base VLM and attach LoRA adapters via PEFT.

        Returns
        -------
        peft.PeftModel
            The LoRA-wrapped model ready for training.
        """
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "peft and transformers are required. "
                "Install with: pip install peft transformers torch"
            ) from exc

        cfg = self.config
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(cfg.base_model.dtype, torch.bfloat16)

        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        # Required so LoRA adapter inputs receive gradients
        base.enable_input_require_grads()

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self._model = get_peft_model(base, lora_cfg)
        self._model.print_trainable_parameters()

        self._processor = AutoProcessor.from_pretrained(
            cfg.base_model.model_name_or_path, trust_remote_code=True
        )
        return self._model

    def prepare_dataset(self, records: list[ImageRecord]) -> Any:
        """Convert a list of :class:`ImageRecord` into a HuggingFace ``Dataset``.

        Only the fields needed at training time (image path, question, answer)
        are included; heavy processing (tokenisation, image loading) is deferred
        to the data collator so it runs in parallel data-loader workers.

        Returns
        -------
        datasets.Dataset
        """
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required. Install with: pip install datasets"
            ) from exc

        rows = [
            {
                "image_path": str(record.image_path),
                "question": record.question,
                "answer": record.answer,
            }
            for record in records
        ]
        return Dataset.from_list(rows)

    def train(
        self,
        train_records: list[ImageRecord],
        val_records: list[ImageRecord],
    ) -> None:
        """Run the full training loop and save the best checkpoint.

        Parameters
        ----------
        train_records:
            Training split as a list of :class:`ImageRecord`.
        val_records:
            Validation split used for best-checkpoint selection.
        """
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        if self._model is None:
            self.prepare_model()

        train_ds = self.prepare_dataset(train_records)
        val_ds = self.prepare_dataset(val_records)

        cfg = self.config
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            report_to=["wandb"],
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=(cfg.base_model.dtype == "float16"),
            bf16=(cfg.base_model.dtype == "bfloat16"),
            logging_steps=10,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=self._make_collate_fn(),
        )
        trainer.train()
        trainer.save_model(cfg.output_dir)

    def merge_and_export(self, output_path: str) -> None:
        """Merge LoRA weights into the base model and save in full-precision.

        Parameters
        ----------
        output_path:
            Directory where the merged model and processor will be saved.
        """
        if self._model is None:
            raise RuntimeError(
                "Call train() or prepare_model() before merge_and_export()."
            )
        merged = self._model.merge_and_unload()
        merged.save_pretrained(output_path)
        if self._processor is not None:
            self._processor.save_pretrained(output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_collate_fn(self) -> Any:
        """Return a data collator that loads images and tokenises batches.

        The collator:
        1. Opens each image from its path.
        2. Formats text as ``"Question: …\\nAnswer: …"``.
        3. Calls the processor with padding and truncation.
        4. Sets ``labels`` equal to ``input_ids`` (standard causal-LM objective).
        """
        processor = self._processor

        def collate_fn(batch: list[dict]) -> dict:
            try:
                from PIL import Image as PILImage
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Pillow is required for image loading.") from exc

            images = []
            texts = []
            for item in batch:
                img = PILImage.open(item["image_path"]).convert("RGB")
                images.append(img)
                texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")

            encoding = processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoding["labels"] = encoding["input_ids"].clone()
            return encoding

        return collate_fn
