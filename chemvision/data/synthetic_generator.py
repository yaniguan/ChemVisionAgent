"""Synthetic XRD spectrum image generator for testing and demos.

Generates realistic-looking XRD diffractogram PNG images without requiring
real instrument data.  Each image simulates a TiO2 sample heated to a given
temperature, modelling the anatase → rutile phase transition that occurs
around 500 °C.

The grain size is estimated via the Scherrer equation:

    D = K·λ / (β·cos θ)

where K=0.9, λ=1.5406 Å (Cu Kα), and β is the FWHM of the main peak.

Usage
-----
>>> from chemvision.data.synthetic_generator import XRDImageGenerator
>>> gen = XRDImageGenerator()
>>> paths = gen.generate_temperature_series(
...     temperatures=[200, 400, 600],
...     output_dir=Path("data/xrd_series"),
... )

No external dependencies beyond ``matplotlib`` and ``numpy``, both of which
are already in the project's dependency list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_CU_KA = 1.5406  # Å  — Cu Kα X-ray wavelength
_SCHERRER_K = 0.9  # dimensionless shape factor


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------


class _Peak(NamedTuple):
    two_theta: float  # degrees
    relative_intensity: float  # 0–1
    assignment: str


# Anatase TiO2 reference peaks (JCPDS 21-1272)
_ANATASE_PEAKS: list[_Peak] = [
    _Peak(25.28, 1.00, "101"),
    _Peak(37.80, 0.20, "004"),
    _Peak(48.05, 0.35, "200"),
    _Peak(53.89, 0.20, "105"),
    _Peak(55.06, 0.22, "211"),
    _Peak(62.69, 0.12, "204"),
    _Peak(68.76, 0.15, "116"),
    _Peak(70.31, 0.10, "220"),
    _Peak(75.03, 0.10, "215"),
]

# Rutile TiO2 reference peaks (JCPDS 21-1276)
_RUTILE_PEAKS: list[_Peak] = [
    _Peak(27.45, 1.00, "110"),
    _Peak(36.09, 0.55, "101"),
    _Peak(39.18, 0.12, "200"),
    _Peak(41.23, 0.23, "111"),
    _Peak(44.05, 0.08, "210"),
    _Peak(54.33, 0.55, "211"),
    _Peak(56.64, 0.18, "220"),
    _Peak(62.74, 0.20, "002"),
    _Peak(64.05, 0.30, "310"),
    _Peak(69.01, 0.12, "301"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class XRDSample:
    """Metadata for a single synthetic XRD measurement."""

    temperature_c: float
    anatase_fraction: float  # 0–1
    grain_size_nm: float  # D from Scherrer equation
    dominant_phase: str  # "anatase" | "mixed" | "rutile"
    image_path: Path | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class XRDImageGenerator:
    """Generate synthetic XRD spectrum PNG images for TiO2 phase transition.

    The anatase → rutile transition is modelled as a sigmoid centred at
    ``transition_temp`` (default 500 °C).  Grain size grows linearly with
    temperature following an Arrhenius-inspired trend.

    Parameters
    ----------
    seed:
        Random seed for reproducible noise.
    transition_temp:
        Centre of the sigmoid transition in °C.
    transition_width:
        Width parameter of the sigmoid (°C).  Smaller → sharper transition.
    base_grain_nm:
        Grain size (nm) at room temperature.
    grain_growth_rate:
        Grain size increase per 100 °C (nm).
    """

    def __init__(
        self,
        seed: int = 42,
        transition_temp: float = 500.0,
        transition_width: float = 80.0,
        base_grain_nm: float = 15.0,
        grain_growth_rate: float = 6.0,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._t_trans = transition_temp
        self._t_width = transition_width
        self._base_grain = base_grain_nm
        self._grain_rate = grain_growth_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_temperature_series(
        self,
        temperatures: list[float],
        output_dir: Path | str,
        dpi: int = 120,
    ) -> list[XRDSample]:
        """Generate one XRD PNG per temperature and return metadata.

        Parameters
        ----------
        temperatures:
            List of temperatures in °C.
        output_dir:
            Directory where PNG images will be written (created if absent).
        dpi:
            Output image resolution.

        Returns
        -------
        list[XRDSample]
            One record per temperature, each with :attr:`image_path` set.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        samples: list[XRDSample] = []
        for t in temperatures:
            sample = self._build_sample(t)
            img_path = out / f"xrd_{int(t)}C.png"
            self._render(sample, img_path, dpi=dpi)
            sample.image_path = img_path.resolve()
            samples.append(sample)

        return samples

    def generate_single(
        self,
        temperature_c: float,
        output_dir: Path | str,
        dpi: int = 120,
    ) -> XRDSample:
        """Generate a single XRD image for one temperature."""
        samples = self.generate_temperature_series(
            [temperature_c], output_dir=output_dir, dpi=dpi
        )
        return samples[0]

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------

    def _anatase_fraction(self, temperature_c: float) -> float:
        """Sigmoid from 1 (pure anatase) to 0 (pure rutile)."""
        x = (temperature_c - self._t_trans) / self._t_width
        return float(1.0 / (1.0 + math.exp(x)))

    def _grain_size(self, temperature_c: float, fwhm_deg: float) -> float:
        """Scherrer grain size in nm."""
        fwhm_rad = math.radians(fwhm_deg)
        # Use primary peak of dominant phase; approximate theta from first major peak
        theta_deg = 25.28 / 2.0 if self._anatase_fraction(temperature_c) > 0.5 else 27.45 / 2.0
        cos_theta = math.cos(math.radians(theta_deg))
        return float(_SCHERRER_K * _CU_KA / (fwhm_rad * cos_theta * 10))  # nm

    def _fwhm(self, temperature_c: float) -> float:
        """FWHM decreases (peaks sharpen) as grains grow with temperature."""
        # At room temp, broad peaks (~0.8°); at 700°C, narrow (~0.2°)
        fraction = min(max(temperature_c / 700.0, 0.0), 1.0)
        return 0.80 - 0.60 * fraction

    def _build_sample(self, temperature_c: float) -> XRDSample:
        af = self._anatase_fraction(temperature_c)
        rf = 1.0 - af
        fwhm = self._fwhm(temperature_c)
        grain_nm = self._grain_size(temperature_c, fwhm)

        if af > 0.7:
            phase = "anatase"
        elif rf > 0.7:
            phase = "rutile"
        else:
            phase = "mixed"

        return XRDSample(
            temperature_c=temperature_c,
            anatase_fraction=af,
            grain_size_nm=grain_nm,
            dominant_phase=phase,
            metadata={
                "fwhm_deg": round(fwhm, 4),
                "rutile_fraction": round(rf, 4),
                "transition_temp_c": self._t_trans,
            },
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _gaussian(
        self,
        two_theta: np.ndarray,
        centre: float,
        fwhm: float,
        amplitude: float,
    ) -> np.ndarray:
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        return amplitude * np.exp(-0.5 * ((two_theta - centre) / sigma) ** 2)

    def _render(self, sample: XRDSample, out_path: Path, dpi: int = 120) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for XRD rendering. "
                "Install with: pip install matplotlib"
            ) from exc

        two_theta = np.linspace(10, 80, 2000)
        intensity = np.zeros_like(two_theta)

        af = sample.anatase_fraction
        rf = 1.0 - af
        fwhm = sample.metadata["fwhm_deg"]

        for peak in _ANATASE_PEAKS:
            intensity += self._gaussian(two_theta, peak.two_theta, fwhm, af * peak.relative_intensity)

        for peak in _RUTILE_PEAKS:
            intensity += self._gaussian(two_theta, peak.two_theta, fwhm, rf * peak.relative_intensity)

        # Add low-level background + Poisson noise
        background = 0.02 + 0.01 * np.exp(-0.05 * (two_theta - 10))
        noise = self._rng.poisson(lam=20, size=len(two_theta)).astype(float) / 500.0
        intensity = intensity + background + noise

        # Normalise to 0–1
        intensity /= intensity.max()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(two_theta, intensity, linewidth=0.9, color="#1f4e79")
        ax.set_xlabel("2θ (degrees)", fontsize=11)
        ax.set_ylabel("Intensity (a.u.)", fontsize=11)
        ax.set_title(
            f"XRD Pattern — TiO₂ at {sample.temperature_c:.0f} °C  "
            f"[{sample.dominant_phase}, D≈{sample.grain_size_nm:.1f} nm]",
            fontsize=10,
        )
        ax.set_xlim(10, 80)
        ax.set_ylim(0, 1.12)

        # Annotate dominant phase peaks
        dominant_peaks = _ANATASE_PEAKS if af >= rf else _RUTILE_PEAKS
        phase_label = "A" if af >= rf else "R"
        for peak in dominant_peaks[:4]:
            y_val = float(self._gaussian(np.array([peak.two_theta]), peak.two_theta, fwhm, 1.0)[0])
            if y_val > 0.15:
                ax.annotate(
                    f"{phase_label}({peak.assignment})",
                    xy=(peak.two_theta, y_val * (af if af >= rf else rf) + background[0]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#c00000",
                )

        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
