"""Scherrer equation: grain size from XRD peak broadening.

D = K·λ / (β·cos θ)

where
  K    = Scherrer constant (0.9 for spherical crystallites)
  λ    = X-ray wavelength in Å (1.5406 Å for Cu Kα)
  β    = FWHM of the diffraction peak in radians
  θ    = Bragg angle in radians (half the 2θ position)
  D    = mean crystallite size in Å
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GrainSizeResult:
    """Scherrer grain size estimate for one XRD peak."""

    two_theta_deg: float
    fwhm_deg: float
    wavelength_angstrom: float = 1.5406
    scherrer_k: float = 0.9

    grain_size_angstrom: float | None = None
    grain_size_nm: float | None = None
    valid: bool = False

    def __post_init__(self) -> None:
        self._compute()

    def _compute(self) -> None:
        try:
            theta_rad = math.radians(self.two_theta_deg / 2.0)
            beta_rad = math.radians(self.fwhm_deg)
            if beta_rad <= 0:
                return
            d = (self.scherrer_k * self.wavelength_angstrom) / (beta_rad * math.cos(theta_rad))
            self.grain_size_angstrom = d
            self.grain_size_nm = d / 10.0
            self.valid = True
        except (ValueError, ZeroDivisionError):
            pass


class ScherrerAnalyzer:
    """Compute grain sizes from a list of extracted XRD peaks.

    Example
    -------
    >>> analyzer = ScherrerAnalyzer()
    >>> results = analyzer.analyze_peaks(
    ...     [(25.3, 0.35), (48.0, 0.42)],   # (2θ, FWHM) in degrees
    ... )
    >>> results[0].grain_size_nm
    24.1
    """

    def __init__(
        self,
        wavelength: float = 1.5406,   # Cu Kα, Å
        scherrer_k: float = 0.9,
    ) -> None:
        self.wavelength = wavelength
        self.scherrer_k = scherrer_k

    def analyze_peaks(
        self,
        peaks: list[tuple[float, float]],
    ) -> list[GrainSizeResult]:
        """Compute grain size for each (2θ, FWHM) pair.

        Parameters
        ----------
        peaks:
            List of ``(two_theta_deg, fwhm_deg)`` tuples.  Use data from
            :class:`~chemvision.skills.outputs.Peak` objects:
            ``(peak.position, peak.fwhm)`` when ``spectrum_type=="XRD"``.
        """
        return [
            GrainSizeResult(
                two_theta_deg=tt,
                fwhm_deg=fwhm,
                wavelength_angstrom=self.wavelength,
                scherrer_k=self.scherrer_k,
            )
            for tt, fwhm in peaks
        ]

    def mean_grain_size_nm(self, peaks: list[tuple[float, float]]) -> float | None:
        """Return the mean grain size in nm across all valid peaks."""
        results = [r for r in self.analyze_peaks(peaks) if r.valid and r.grain_size_nm]
        if not results:
            return None
        return sum(r.grain_size_nm for r in results) / len(results)  # type: ignore[misc]
