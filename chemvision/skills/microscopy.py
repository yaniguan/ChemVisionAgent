"""Skill: analyse morphology, particle size, and scale bar in microscopy images."""

from __future__ import annotations

from typing import Any, Literal, get_args

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import (
    MicroscopyAnalysis,
    MorphologyInfo,
    ParticleMeasurement,
    ScaleBar,
    SizeStatistics,
)

_PROMPT_TEMPLATE = """\
You are an expert microscopist specialising in materials characterisation (SEM, TEM, STEM, AFM, optical microscopy).

Imaging context: {imaging_context}

Carefully examine the image and extract the following information.

STEP 1 — Scale bar
  Find the scale bar (usually in a corner). Record its numeric value, unit, and pixel length.
  Use the scale bar to calibrate particle-size measurements.
  If no scale bar is visible, set scale_bar to null.

STEP 2 — Imaging modality & magnification
  Identify the instrument type (SEM / TEM / STEM / AFM / OM / other) from any text overlays,
  detector labels, or image appearance. Record the magnification label if shown.

STEP 3 — Morphology
  Describe the dominant particle/grain shape, surface texture, and aggregation state.
  Use one of these controlled values where possible:
    shape:           spherical | rod | platelet | dendritic | porous | irregular | core-shell | other
    surface_texture: smooth | rough | faceted | porous | other
    aggregation:     dispersed | agglomerated | sintered | clustered | other

STEP 4 — Individual particle measurements (up to 20 representative particles)
  For each measured particle record:
    • diameter (longest axis) in the calibrated unit
    • aspect_ratio (length / width; 1.0 for a sphere)
    • shape (same vocabulary as above)
    • normalised image coordinates (location_x, location_y) in [0, 1]

STEP 5 — Size statistics
  Compute (or estimate) mean, std, min, max diameter across all measured particles.
  Choose distribution: monodisperse (<10 % CV) | polydisperse | bimodal | unknown

Your output MUST be a single valid JSON object:
{{
  "scale_bar": {{
    "value": <number or null>,
    "unit": "<nm|μm|mm|other>",
    "pixel_length": <integer or null>,
    "nm_per_pixel": <number or null>
  }},
  "imaging_modality": "<SEM|TEM|STEM|AFM|OM|other>",
  "magnification": "<e.g. 50000x, or null>",
  "morphology": {{
    "shape": "<spherical|rod|platelet|dendritic|porous|irregular|core-shell|other>",
    "surface_texture": "<smooth|rough|faceted|porous|other>",
    "aggregation": "<dispersed|agglomerated|sintered|clustered|other>",
    "description": "<1–3 sentence free-text summary>"
  }},
  "particles": [
    {{
      "diameter": <number in calibrated unit>,
      "aspect_ratio": <number>,
      "shape": "<shape>",
      "location_x": <0.0–1.0>,
      "location_y": <0.0–1.0>
    }}
  ],
  "size_statistics": {{
    "mean_diameter": <number>,
    "std_diameter": <number>,
    "min_diameter": <number>,
    "max_diameter": <number>,
    "unit": "<nm|μm|mm>",
    "distribution": "<monodisperse|polydisperse|bimodal|unknown>",
    "particle_count": <integer>
  }},
  "confidence": <0.0–1.0>
}}

Important:
- If a scale bar is present, use it to report all sizes in calibrated units.
- If no scale bar is present, report sizes in pixels and set unit to "px".
- Respond with only the JSON object, no other text.
"""

_VALID_SHAPES: frozenset[str] = frozenset(
    ["spherical", "rod", "platelet", "dendritic", "porous", "irregular", "core-shell", "other"]
)
_VALID_TEXTURES: frozenset[str] = frozenset(["smooth", "rough", "faceted", "porous", "other"])
_VALID_AGGREGATIONS: frozenset[str] = frozenset(
    ["dispersed", "agglomerated", "sintered", "clustered", "other"]
)
_VALID_DISTRIBUTIONS: frozenset[str] = frozenset(
    get_args(Literal["monodisperse", "polydisperse", "bimodal", "unknown"])
)
_VALID_MODALITIES: frozenset[str] = frozenset(["SEM", "TEM", "STEM", "AFM", "OM", "other"])


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class MicroscopySkill(BaseSkill):
    """Analyse morphology, particle size distribution, and scale bar in microscopy images.

    Handles SEM, TEM, STEM, AFM, and optical microscopy images.  Returns a
    structured :class:`~chemvision.skills.outputs.MicroscopyAnalysis` with:

    - :attr:`morphology` — dominant shape, surface texture, aggregation state
    - :attr:`particles`  — per-particle diameter, aspect ratio, position (up to 20)
    - :attr:`size_statistics` — mean / std / min / max diameter, distribution type
    - :attr:`scale_bar`  — calibrated value, unit, and pixel length read from image
    - :attr:`imaging_modality` — SEM / TEM / STEM / AFM / OM / other

    Example
    -------
    >>> skill = MicroscopySkill()
    >>> result = skill(image, model, imaging_context="SEM image of ZnO nanoparticles")
    >>> result.morphology.shape
    'spherical'
    >>> result.size_statistics.mean_diameter
    45.3
    >>> result.scale_bar.unit
    'nm'
    """

    name = "analyze_microscopy"

    def build_prompt(self, **kwargs: Any) -> str:
        imaging_context = to_str(
            kwargs.get("imaging_context"),
            "microscopy image of a material sample",
        )
        return _PROMPT_TEMPLATE.format(imaging_context=imaging_context)

    def __call__(
        self,
        image: Image,
        model: Any,
        **kwargs: Any,
    ) -> MicroscopyAnalysis:
        """Run microscopy analysis and return a typed :class:`MicroscopyAnalysis`.

        Parameters
        ----------
        image:
            RGB PIL image from SEM / TEM / AFM / optical microscope.
        model:
            Loaded vision model (local or
            :class:`~chemvision.agent.adapter.AnthropicVisionFallback`).
        imaging_context:
            Brief description of the sample and instrument, e.g.
            ``"SEM image of ZnO nanoparticles on Si substrate"`` (kwarg).
        """
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        # --- Scale bar -------------------------------------------------------
        scale_bar: ScaleBar | None = None
        sb_data = data.get("scale_bar")
        if isinstance(sb_data, dict) and sb_data.get("value") is not None:
            pl_raw = sb_data.get("pixel_length")
            scale_bar = ScaleBar(
                raw_output=raw,
                value=to_float(sb_data.get("value")),
                unit=to_str(sb_data.get("unit"), "nm"),
                pixel_length=int(pl_raw) if pl_raw is not None else None,
                nm_per_pixel=to_float(sb_data.get("nm_per_pixel")),
            )

        # --- Imaging modality ------------------------------------------------
        raw_modality = to_str(data.get("imaging_modality"), "other").upper()
        imaging_modality = raw_modality if raw_modality in _VALID_MODALITIES else "other"
        magnification = to_str(data.get("magnification")) or None

        # --- Morphology ------------------------------------------------------
        morphology: MorphologyInfo | None = None
        morph_data = data.get("morphology")
        if isinstance(morph_data, dict):
            raw_shape = to_str(morph_data.get("shape"), "other").lower()
            raw_tex = to_str(morph_data.get("surface_texture"), "other").lower()
            raw_agg = to_str(morph_data.get("aggregation"), "other").lower()
            morphology = MorphologyInfo(
                raw_output=raw,
                shape=raw_shape if raw_shape in _VALID_SHAPES else "other",
                surface_texture=raw_tex if raw_tex in _VALID_TEXTURES else "other",
                aggregation=raw_agg if raw_agg in _VALID_AGGREGATIONS else "other",
                description=to_str(morph_data.get("description")),
            )

        # --- Individual particles --------------------------------------------
        particles: list[ParticleMeasurement] = []
        for p in to_list(data.get("particles")):
            if not isinstance(p, dict):
                continue
            particles.append(
                ParticleMeasurement(
                    raw_output=raw,
                    diameter=to_float(p.get("diameter")),
                    aspect_ratio=to_float(p.get("aspect_ratio")),
                    shape=to_str(p.get("shape"), "unknown"),
                    location_x=_clamp01(float(p.get("location_x") or 0.0)),
                    location_y=_clamp01(float(p.get("location_y") or 0.0)),
                )
            )

        # --- Size statistics -------------------------------------------------
        size_statistics: SizeStatistics | None = None
        stats_data = data.get("size_statistics")
        if isinstance(stats_data, dict):
            raw_dist = to_str(stats_data.get("distribution"), "unknown").lower()
            distribution = raw_dist if raw_dist in _VALID_DISTRIBUTIONS else "unknown"
            count_raw = stats_data.get("particle_count")
            size_statistics = SizeStatistics(
                raw_output=raw,
                mean_diameter=to_float(stats_data.get("mean_diameter")),
                std_diameter=to_float(stats_data.get("std_diameter")),
                min_diameter=to_float(stats_data.get("min_diameter")),
                max_diameter=to_float(stats_data.get("max_diameter")),
                unit=to_str(stats_data.get("unit"), "nm"),
                distribution=distribution,
                particle_count=int(count_raw) if count_raw is not None else None,
            )

        return MicroscopyAnalysis(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            morphology=morphology,
            particles=particles,
            size_statistics=size_statistics,
            scale_bar=scale_bar,
            imaging_modality=imaging_modality,
            magnification=magnification,
        )
