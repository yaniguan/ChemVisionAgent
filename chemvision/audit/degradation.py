"""DegradationTester: binary-search for the critical robustness envelope."""

from __future__ import annotations

import io
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL.Image import Image
from pydantic import BaseModel, Field

from chemvision.data.schema import ImageRecord
from chemvision.models.base import BaseVisionModel


# ---------------------------------------------------------------------------
# Config + result schemas
# ---------------------------------------------------------------------------


class DegradationConfig(BaseModel):
    """Configuration for :class:`DegradationTester`."""

    threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Accuracy threshold that defines the critical point."
    )
    n_samples_per_eval: int = Field(
        20, gt=0, description="Records sampled per binary-search accuracy estimate."
    )
    n_binary_search_iters: int = Field(
        8, gt=0, description="Binary search iterations per degradation type."
    )
    seed: int = Field(42)
    output_dir: Path = Field(Path("reports/"), description="Directory for JSON envelope output.")
    score_fn: str = Field(
        "substring",
        description="Answer scoring strategy forwarded to accuracy evaluation.",
    )


@dataclass
class DegradationResult:
    """Outcome of the binary search for one degradation type."""

    degradation_type: str
    param_name: str
    param_unit: str
    critical_param: float
    accuracy_at_critical: float
    param_clean: float
    param_max_degradation: float
    n_model_calls: int

    @property
    def normalized_tolerance(self) -> float:
        """Critical param normalised to [0, 1] where 1 = maximum tolerance."""
        span = abs(self.param_max_degradation - self.param_clean)
        if span == 0:
            return 0.0
        return abs(self.critical_param - self.param_clean) / span

    @property
    def robustness_label(self) -> str:
        t = self.normalized_tolerance
        if t >= 0.6:
            return "high"
        if t >= 0.3:
            return "moderate"
        return "low"


@dataclass
class ReliabilityEnvelope:
    """Full reliability envelope for a model under multiple degradations."""

    model_name: str
    threshold: float
    results: dict[str, DegradationResult]
    evaluated_at: str = ""

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def save_json(self, path: Path) -> Path:
        """Serialise to *path* (created/overwritten) and return it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "evaluated_at": self.evaluated_at,
            "degradations": {
                k: asdict(v) for k, v in self.results.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    @classmethod
    def load_json(cls, path: Path) -> ReliabilityEnvelope:
        """Load a previously saved envelope from *path*."""
        raw = json.loads(Path(path).read_text())
        results = {
            k: DegradationResult(**v) for k, v in raw["degradations"].items()
        }
        return cls(
            model_name=raw["model_name"],
            threshold=raw["threshold"],
            results=results,
            evaluated_at=raw.get("evaluated_at", ""),
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DegradationTester:
    """Probe robustness to five canonical image degradations.

    For each degradation type, a binary search identifies the critical
    degradation parameter at which the model's accuracy drops below
    :attr:`DegradationConfig.threshold` (default 0.7).  The result is
    exported as a JSON "reliability envelope".

    Degradation types
    -----------------
    gaussian_noise
        Additive Gaussian noise; parameter = standard deviation σ (0–100).
    jpeg_compression
        JPEG re-encoding; parameter = quality (95 = clean → 1 = max loss).
    occlusion
        Random rectangular black patch; parameter = fraction of area (0–0.9).
    downsampling
        Bicubic downsample + upsample; parameter = scale factor (1.0–0.1).
    color_shift
        Constant channel-wise brightness shift; parameter = magnitude (0–100).

    Example
    -------
    >>> cfg = DegradationConfig(threshold=0.7)
    >>> tester = DegradationTester(cfg)
    >>> envelope = tester.run(model, dataset)
    >>> envelope.results["gaussian_noise"].robustness_label
    'moderate'
    >>> envelope.save_json(Path("reports/envelope.json"))
    """

    # (param_name, param_unit, clean_value, max_degradation_value)
    _DEGRADATION_SPECS: dict[str, tuple[str, str, float, float]] = {
        "gaussian_noise":   ("sigma",        "px intensity", 0.0,  100.0),
        "jpeg_compression": ("quality",      "JPEG quality", 95.0, 1.0),
        "occlusion":        ("fraction",     "area fraction", 0.0, 0.9),
        "downsampling":     ("scale_factor", "relative scale", 1.0, 0.1),
        "color_shift":      ("magnitude",    "px intensity", 0.0,  100.0),
    }

    def __init__(self, config: DegradationConfig | None = None) -> None:
        self.config = config or DegradationConfig()
        self._model_call_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        model: BaseVisionModel,
        dataset: list[ImageRecord],
    ) -> ReliabilityEnvelope:
        """Run binary-search for all five degradation types.

        Parameters
        ----------
        model:
            Loaded vision-language model.
        dataset:
            Evaluation records (images + Q&A pairs).

        Returns
        -------
        ReliabilityEnvelope
            Critical parameters and robustness labels for each degradation.
        """
        results: dict[str, DegradationResult] = {}

        apply_fns: dict[str, Callable[[Image, float], Image]] = {
            "gaussian_noise":   self._apply_gaussian_noise,
            "jpeg_compression": self._apply_jpeg_compression,
            "occlusion":        self._apply_occlusion,
            "downsampling":     self._apply_downsampling,
            "color_shift":      self._apply_color_shift,
        }

        for deg_type, (param_name, param_unit, clean, maxdeg) in self._DEGRADATION_SPECS.items():
            self._model_call_count = 0
            critical, acc = self._binary_search(
                apply_fn=apply_fns[deg_type],
                param_clean=clean,
                param_degraded=maxdeg,
                model=model,
                records=dataset,
            )
            results[deg_type] = DegradationResult(
                degradation_type=deg_type,
                param_name=param_name,
                param_unit=param_unit,
                critical_param=critical,
                accuracy_at_critical=acc,
                param_clean=clean,
                param_max_degradation=maxdeg,
                n_model_calls=self._model_call_count,
            )

        envelope = ReliabilityEnvelope(
            model_name=getattr(model, "config", None) and model.config.model_name_or_path  # type: ignore[union-attr]
            or "unknown",
            threshold=self.config.threshold,
            results=results,
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        envelope.save_json(out_dir / "reliability_envelope.json")
        return envelope

    # ------------------------------------------------------------------
    # Degradation application methods (each returns a PIL Image)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_gaussian_noise(image: Image, sigma: float) -> Image:
        """Add zero-mean Gaussian noise with standard deviation *sigma*."""
        import numpy as np
        from PIL import Image as PILImage

        arr = np.array(image).astype(np.float32)
        rng = np.random.default_rng(seed=0)
        noise = rng.normal(0.0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return PILImage.fromarray(noisy)

    @staticmethod
    def _apply_jpeg_compression(image: Image, quality: float) -> Image:
        """Re-encode as JPEG at *quality* and decode back to RGB."""
        from PIL import Image as PILImage

        q = max(1, min(95, int(round(quality))))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return PILImage.open(buf).copy()

    @staticmethod
    def _apply_occlusion(image: Image, fraction: float) -> Image:
        """Fill a random rectangular patch covering *fraction* of the image."""
        import numpy as np
        from PIL import Image as PILImage

        arr = np.array(image).copy()
        h, w = arr.shape[:2]
        area = fraction * h * w
        # Square patch whose area ≈ target fraction
        side = int(math.sqrt(area))
        side = min(side, h, w)
        if side <= 0:
            return image.copy()
        rng = np.random.default_rng(seed=1)
        y0 = int(rng.integers(0, max(1, h - side)))
        x0 = int(rng.integers(0, max(1, w - side)))
        arr[y0 : y0 + side, x0 : x0 + side] = 0
        return PILImage.fromarray(arr)

    @staticmethod
    def _apply_downsampling(image: Image, scale: float) -> Image:
        """Downsample by *scale* then upsample back to original size."""
        from PIL import Image as PILImage

        scale = max(0.05, min(1.0, scale))
        w, h = image.size
        small_w, small_h = max(1, int(w * scale)), max(1, int(h * scale))
        small = image.resize((small_w, small_h), PILImage.BICUBIC)
        return small.resize((w, h), PILImage.BICUBIC)

    @staticmethod
    def _apply_color_shift(image: Image, magnitude: float) -> Image:
        """Add a constant per-channel brightness offset of *magnitude*."""
        import numpy as np
        from PIL import Image as PILImage

        arr = np.array(image).astype(np.float32)
        # Deterministic pseudo-random shift per channel
        rng = np.random.default_rng(seed=2)
        shifts = rng.uniform(-magnitude, magnitude, (1, 1, arr.shape[2]))
        shifted = np.clip(arr + shifts, 0, 255).astype(np.uint8)
        return PILImage.fromarray(shifted)

    # ------------------------------------------------------------------
    # Binary search
    # ------------------------------------------------------------------

    def _binary_search(
        self,
        apply_fn: Callable[[Image, float], Image],
        param_clean: float,
        param_degraded: float,
        model: BaseVisionModel,
        records: list[ImageRecord],
    ) -> tuple[float, float]:
        """Find the critical parameter where accuracy first drops below threshold.

        Uses a normalised [0, 1] search variable *t* where t=0 is the clean
        image and t=1 is maximum degradation.  The actual parameter is
        interpolated: ``param = param_clean + t * (param_degraded - param_clean)``.

        Returns
        -------
        tuple[float, float]
            ``(critical_param, accuracy_at_critical)``
        """
        t_low, t_high = 0.0, 1.0

        for _ in range(self.config.n_binary_search_iters):
            t_mid = (t_low + t_high) / 2.0
            param = param_clean + t_mid * (param_degraded - param_clean)
            acc = self._evaluate_accuracy(model, records, apply_fn, param)
            if acc >= self.config.threshold:
                t_low = t_mid   # can tolerate more degradation
            else:
                t_high = t_mid  # too much degradation

        t_critical = (t_low + t_high) / 2.0
        critical_param = param_clean + t_critical * (param_degraded - param_clean)
        final_acc = self._evaluate_accuracy(model, records, apply_fn, critical_param)
        return critical_param, final_acc

    def _evaluate_accuracy(
        self,
        model: BaseVisionModel,
        records: list[ImageRecord],
        apply_fn: Callable[[Image, float], Image],
        param: float,
    ) -> float:
        """Sample *n_samples_per_eval* records, apply degradation, score answers."""
        from PIL import Image as PILImage

        rng = random.Random(self.config.seed)
        n = min(self.config.n_samples_per_eval, len(records))
        sample = rng.sample(records, n)

        num_correct = 0
        for record in sample:
            image = PILImage.open(record.image_path).convert("RGB")
            degraded = apply_fn(image, param)
            predicted = model.generate(degraded, record.question)
            if self._score_answer(predicted, record.answer):
                num_correct += 1
            self._model_call_count += 1

        return num_correct / n if n > 0 else 0.0

    @staticmethod
    def _score_answer(predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()
        return gt in pred
