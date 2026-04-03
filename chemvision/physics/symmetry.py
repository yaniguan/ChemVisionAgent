"""Crystal symmetry analysis via spglib.

Given lattice parameters or a full structure (cell + fractional positions +
species), this module determines the space group, crystal system, and
Wyckoff positions using the spglib C library.

This acts as a hard physics constraint on any structure extracted by
the vision skills: if a proposed lattice does not have a valid space group
the downstream prediction pipeline can flag it as unphysical.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import spglib  # already installed

logger = logging.getLogger(__name__)


@dataclass
class SymmetryResult:
    """Output of spglib symmetry analysis."""

    space_group_number: int | None = None
    space_group_symbol: str = ""
    crystal_system: str = ""          # cubic, hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, triclinic
    point_group: str = ""
    hall_symbol: str = ""
    wyckoff_letters: list[str] = field(default_factory=list)
    equivalent_atoms: list[int] = field(default_factory=list)
    precision: float = 1e-5
    is_valid: bool = False

    @property
    def summary(self) -> str:
        if not self.is_valid:
            return "Symmetry analysis failed"
        return (
            f"{self.crystal_system.capitalize()} | "
            f"{self.space_group_symbol} (#{self.space_group_number}) | "
            f"Point group {self.point_group}"
        )


_CRYSTAL_SYSTEM: dict[str, str] = {
    "1": "triclinic", "2": "triclinic",
    "3": "monoclinic", **{str(i): "monoclinic" for i in range(3, 16)},
    **{str(i): "orthorhombic" for i in range(16, 75)},
    **{str(i): "tetragonal" for i in range(75, 143)},
    **{str(i): "trigonal" for i in range(143, 168)},
    **{str(i): "hexagonal" for i in range(168, 195)},
    **{str(i): "cubic" for i in range(195, 231)},
}


class CrystalSymmetryAnalyzer:
    """Determine symmetry of a crystal structure using spglib.

    Example
    -------
    Analyse a face-centred cubic Cu lattice:

    >>> analyzer = CrystalSymmetryAnalyzer()
    >>> lattice = [[3.615, 0, 0], [0, 3.615, 0], [0, 0, 3.615]]
    >>> positions = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    >>> numbers = [29, 29, 29, 29]
    >>> result = analyzer.analyze(lattice, positions, numbers)
    >>> result.space_group_symbol
    'Fm-3m'
    >>> result.crystal_system
    'cubic'
    """

    def __init__(self, symprec: float = 1e-5, angle_tolerance: float = -1.0) -> None:
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def analyze(
        self,
        lattice: list[list[float]] | np.ndarray,
        positions: list[list[float]] | np.ndarray,
        atomic_numbers: list[int] | np.ndarray,
    ) -> SymmetryResult:
        """Run spglib symmetry detection.

        Parameters
        ----------
        lattice:
            3×3 matrix where rows are lattice vectors in Å.
        positions:
            (N, 3) array of fractional atomic positions.
        atomic_numbers:
            Length-N list of atomic numbers (e.g. [22, 8, 8] for TiO2).
        """
        cell = (
            np.array(lattice, dtype=float),
            np.array(positions, dtype=float),
            np.array(atomic_numbers, dtype=int),
        )
        result = SymmetryResult(precision=self.symprec)
        try:
            dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec)
            if dataset is None:
                return result
            # spglib >= 2.5 returns a SpglibDataset namedtuple-like object
            if hasattr(dataset, "number"):
                sgn = int(dataset.number)
            else:
                sgn = int(dataset["number"])

            result.space_group_number = sgn
            result.crystal_system = _CRYSTAL_SYSTEM.get(str(sgn), "unknown")

            sg_type = spglib.get_spacegroup_type(sgn)
            if sg_type:
                result.space_group_symbol = sg_type.get("international_short", "")
                result.hall_symbol = sg_type.get("hall_symbol", "")
                result.point_group = sg_type.get("pointgroup_international", "")

            if hasattr(dataset, "wyckoffs"):
                result.wyckoff_letters = list(dataset.wyckoffs) if dataset.wyckoffs is not None else []
            if hasattr(dataset, "equivalent_atoms"):
                result.equivalent_atoms = (
                    dataset.equivalent_atoms.tolist()
                    if dataset.equivalent_atoms is not None
                    else []
                )
            result.is_valid = True
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Symmetry analysis failed: %s", exc)
        return result

    def from_lattice_params(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        species: list[int] | None = None,
        fractional_positions: list[list[float]] | None = None,
    ) -> SymmetryResult:
        """Build a cell from conventional lattice parameters and run analysis.

        Parameters
        ----------
        a, b, c:
            Lattice lengths in Å.
        alpha, beta, gamma:
            Inter-axial angles in degrees.
        species:
            Atomic numbers for each site (default: all Si).
        fractional_positions:
            Fractional coordinates; defaults to a single atom at [0,0,0].
        """
        lattice = self._params_to_matrix(a, b, c, alpha, beta, gamma)
        if fractional_positions is None:
            fractional_positions = [[0.0, 0.0, 0.0]]
        if species is None:
            species = [14] * len(fractional_positions)  # Si placeholder
        return self.analyze(lattice, fractional_positions, species)

    @staticmethod
    def _params_to_matrix(
        a: float, b: float, c: float,
        alpha: float, beta: float, gamma: float,
    ) -> np.ndarray:
        """Convert conventional lattice parameters to a 3×3 lattice matrix."""
        alpha_r, beta_r, gamma_r = map(math.radians, (alpha, beta, gamma))
        cx = math.cos(beta_r)
        cy = (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / math.sin(gamma_r)
        cz = math.sqrt(max(1.0 - cx**2 - cy**2, 0.0))
        return np.array([
            [a, 0.0, 0.0],
            [b * math.cos(gamma_r), b * math.sin(gamma_r), 0.0],
            [c * cx, c * cy, c * cz],
        ])
