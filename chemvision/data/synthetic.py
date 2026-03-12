"""Synthetic data generator for crystallographic and molecular-dynamics images.

Given a VASP OUTCAR or LAMMPS dump file this module:

1. Parses the atomic structure with ASE (auto-detects format).
2. Renders a three-panel orthogonal-projection PNG via matplotlib + ase.visualize.
3. Auto-generates QA pairs at three difficulty levels from ground-truth parameters
   (lattice constants, atom counts, energy, forces, density …).

Heavy optional dependencies (``ase``, ``matplotlib``) are imported lazily so that
the rest of the package remains importable even when those libraries are absent.

Requires
--------
ase>=3.23, matplotlib>=3.8, numpy>=1.26
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import numpy as np

from chemvision.data.schema import ImageDomain, ImageRecord


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class QAPair(NamedTuple):
    """A single question–answer pair derived from ground-truth parameters."""

    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"]


@dataclass
class ParsedStructure:
    """Parsed crystallographic or MD structure wrapping an ASE Atoms object.

    The ``atoms`` field holds an ``ase.Atoms`` instance.  All derived
    properties delegate to it so the rest of the module stays type-clean.
    """

    atoms: Any  # ase.Atoms — Any to avoid hard import at module-load time
    source_format: Literal["vasp", "lammps"]
    source_path: Path
    total_energy: float | None = None  # eV (from DFT/MD)
    forces: np.ndarray | None = None  # shape (N, 3), eV/Å
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Eagerly cache forces from the ASE object if not supplied explicitly."""
        if self.forces is None:
            try:
                f = self.atoms.get_forces()
                if f is not None and len(f) > 0:
                    self.forces = np.asarray(f)
            except Exception:
                pass

    # ---- derived properties -----------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def chemical_formula(self) -> str:
        return self.atoms.get_chemical_formula()

    @property
    def atom_symbols(self) -> list[str]:
        return list(self.atoms.get_chemical_symbols())

    @property
    def atom_counts(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(self.atom_symbols))

    @property
    def unique_elements(self) -> list[str]:
        return sorted(set(self.atom_symbols))

    @property
    def lattice_vectors(self) -> np.ndarray:
        """3×3 matrix of lattice vectors in Å."""
        return np.asarray(self.atoms.cell[:])

    @property
    def lattice_constants(self) -> tuple[float, float, float]:
        """(a, b, c) in Å."""
        norms = np.linalg.norm(self.lattice_vectors, axis=1)
        return (float(norms[0]), float(norms[1]), float(norms[2]))

    @property
    def lattice_angles(self) -> tuple[float, float, float]:
        """(α, β, γ) in degrees."""
        return tuple(float(x) for x in self.atoms.cell.angles())  # type: ignore[return-value]

    @property
    def volume(self) -> float:
        """Unit-cell volume in Å³."""
        return float(self.atoms.get_volume())

    @property
    def density(self) -> float:
        """Mass density in g/cm³.  1 amu/Å³ = 1.66054 g/cm³."""
        total_mass = float(self.atoms.get_masses().sum())
        return total_mass * 1.66054 / self.volume

    @property
    def has_forces(self) -> bool:
        return self.forces is not None and len(self.forces) > 0

    @property
    def mean_force_magnitude(self) -> float | None:
        if not self.has_forces:
            return None
        return float(np.mean(np.linalg.norm(self.forces, axis=1)))  # type: ignore[arg-type]

    @property
    def max_force_magnitude(self) -> float | None:
        if not self.has_forces:
            return None
        return float(np.max(np.linalg.norm(self.forces, axis=1)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lattice classification helper (module-level so tests can import it directly)
# ---------------------------------------------------------------------------


def classify_bravais(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> str:
    """Heuristic Bravais lattice classification from lattice parameters.

    Parameters
    ----------
    a, b, c:
        Lattice constants in Å.
    alpha, beta, gamma:
        Lattice angles in degrees.

    Returns
    -------
    str
        One of ``'cubic'``, ``'tetragonal'``, ``'orthorhombic'``,
        ``'hexagonal'``, or ``'triclinic / monoclinic'``.
    """
    len_tol = 0.05  # Å
    ang_tol = 1.0  # degrees

    right_angles = all(abs(x - 90.0) < ang_tol for x in (alpha, beta, gamma))
    ab_equal = abs(a - b) < len_tol
    bc_equal = abs(b - c) < len_tol

    if ab_equal and bc_equal and right_angles:
        return "cubic"
    if ab_equal and not bc_equal and right_angles:
        return "tetragonal"
    if right_angles:
        return "orthorhombic"
    if ab_equal and abs(alpha - 90.0) < ang_tol and abs(abs(gamma) - 120.0) < ang_tol:
        return "hexagonal"
    return "triclinic / monoclinic"


# ---------------------------------------------------------------------------
# QA template builder
# ---------------------------------------------------------------------------


def _build_template_pool(
    structure: ParsedStructure,
) -> list[tuple[str, Callable[[ParsedStructure], str], Literal["easy", "medium", "hard"]]]:
    """Return (question, answer_fn, difficulty) triples for *structure*.

    Templates that require optional data (forces, energy) are included only
    when that data is actually present in the structure.
    """
    templates: list[tuple[str, Callable[[ParsedStructure], str], Any]] = [
        # ---- easy -----------------------------------------------------------
        (
            "How many atoms are in this crystal structure?",
            lambda s: str(s.n_atoms),
            "easy",
        ),
        (
            "What is the chemical formula of this structure?",
            lambda s: s.chemical_formula,
            "easy",
        ),
        (
            "How many distinct element types are present in this structure?",
            lambda s: str(len(s.unique_elements)),
            "easy",
        ),
        (
            "Which chemical elements are present in this structure?",
            lambda s: ", ".join(s.unique_elements),
            "easy",
        ),
        # ---- medium ---------------------------------------------------------
        (
            "What are the lattice constants a, b, and c of this structure in Ångströms?",
            lambda s: (
                f"a = {s.lattice_constants[0]:.3f} Å, "
                f"b = {s.lattice_constants[1]:.3f} Å, "
                f"c = {s.lattice_constants[2]:.3f} Å"
            ),
            "medium",
        ),
        (
            "What is the unit cell volume of this structure in cubic Ångströms?",
            lambda s: f"{s.volume:.3f} Å³",
            "medium",
        ),
        (
            "What is the mass density of this material in g/cm³?",
            lambda s: f"{s.density:.4f} g/cm³",
            "medium",
        ),
        # ---- hard -----------------------------------------------------------
        (
            "Based on the lattice constants and angles, what is the Bravais lattice type "
            "of this structure?",
            lambda s: classify_bravais(*s.lattice_constants, *s.lattice_angles),
            "hard",
        ),
    ]

    # Per-element atom-count questions (easy)
    for elem, count in structure.atom_counts.items():
        templates.append((
            f"How many {elem} atoms are present in this structure?",
            # Default args capture loop variables correctly
            lambda s, _e=elem, _c=count: str(_c),
            "easy",
        ))

    # Energy-dependent questions (medium) — only if DFT energy is available
    if structure.total_energy is not None:
        templates.extend([
            (
                "What is the total DFT energy of this structure in eV?",
                lambda s: f"{s.total_energy:.4f} eV",  # type: ignore[union-attr]
                "medium",
            ),
            (
                "What is the DFT energy per atom of this structure in eV/atom?",
                lambda s: f"{s.total_energy / s.n_atoms:.4f} eV/atom",  # type: ignore[operator]
                "medium",
            ),
        ])

    # Force-dependent questions (hard) — only if forces are available
    if structure.has_forces:
        templates.extend([
            (
                "What is the mean force magnitude on all atoms in eV/Å?",
                lambda s: f"{s.mean_force_magnitude:.4f} eV/Å",  # type: ignore[union-attr]
                "hard",
            ),
            (
                "What is the maximum atomic force magnitude in eV/Å?",
                lambda s: f"{s.max_force_magnitude:.4f} eV/Å",  # type: ignore[union-attr]
                "hard",
            ),
        ])

    return templates  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SyntheticGenerator:
    """Generate :class:`~chemvision.data.schema.ImageRecord` objects from
    VASP OUTCAR or LAMMPS dump files.

    Each input file produces **one rendered image** and **multiple QA pairs**
    (one :class:`~chemvision.data.schema.ImageRecord` per QA pair, all
    pointing to the same image).

    Example
    -------
    >>> gen = SyntheticGenerator(seed=42)
    >>> records = gen.generate(Path("OUTCAR"), output_dir=Path("data/raw/synthetic"))
    >>> len(records)
    6
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        file_path: Path,
        output_dir: Path,
        n_questions_per_difficulty: int = 2,
    ) -> list[ImageRecord]:
        """Parse a simulation file and return annotated :class:`ImageRecord` objects.

        Parameters
        ----------
        file_path:
            Path to a VASP OUTCAR or LAMMPS dump file.
        output_dir:
            Directory where rendered PNG images will be saved (created if absent).
        n_questions_per_difficulty:
            How many QA pairs to generate per difficulty tier (easy / medium / hard).

        Returns
        -------
        list[ImageRecord]
            One record per generated question, all sharing the same image path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        structure = self._parse(file_path)
        image_path = self._render(structure, output_dir)
        qa_pairs = self._generate_qa_pairs(structure, n_per_difficulty=n_questions_per_difficulty)

        records: list[ImageRecord] = []
        for i, qa in enumerate(qa_pairs):
            records.append(
                ImageRecord(
                    id=f"{file_path.stem}_q{i:03d}",
                    image_path=image_path,
                    domain=ImageDomain.CRYSTAL_STRUCTURE,
                    question=qa.question,
                    answer=qa.answer,
                    difficulty=qa.difficulty,
                    source=f"synthetic_{structure.source_format}",
                    metadata={
                        "source_file": str(file_path),
                        "chemical_formula": structure.chemical_formula,
                        "n_atoms": structure.n_atoms,
                        "source_format": structure.source_format,
                        "volume_ang3": round(structure.volume, 4),
                    },
                )
            )
        return records

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, file_path: Path) -> ParsedStructure:
        """Auto-detect format and dispatch to the appropriate parser."""
        fmt = self.detect_format(file_path)
        if fmt == "vasp":
            return self.parse_vasp_outcar(file_path)
        return self.parse_lammps_dump(file_path)

    @staticmethod
    def detect_format(path: Path) -> Literal["vasp", "lammps"]:
        """Heuristic format detection from filename and first line of content.

        Checks (in order):
        1. Filename contains ``OUTCAR`` / ``POSCAR`` / ``CONTCAR`` → vasp
        2. Filename contains ``dump`` → lammps
        3. First line contains ``ITEM:`` → lammps
        4. Fallback → vasp
        """
        name = path.name.upper()
        if any(kw in name for kw in ("OUTCAR", "POSCAR", "CONTCAR")):
            return "vasp"
        if "dump" in path.name.lower():
            return "lammps"
        try:
            with path.open() as fh:
                first_line = fh.readline()
            if "ITEM:" in first_line:
                return "lammps"
        except OSError:
            pass
        return "vasp"

    @staticmethod
    def parse_vasp_outcar(path: Path) -> ParsedStructure:
        """Parse the final ionic step from a VASP OUTCAR file using ASE.

        Parameters
        ----------
        path:
            Path to the OUTCAR file.

        Returns
        -------
        ParsedStructure

        Raises
        ------
        ImportError
            If ``ase`` is not installed.
        """
        try:
            import ase.io
        except ImportError as exc:
            raise ImportError(
                "ase is required for VASP parsing. Install with: pip install ase"
            ) from exc

        atoms = ase.io.read(str(path), index=-1, format="vasp-out")

        energy: float | None = None
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            pass

        forces: np.ndarray | None = None
        try:
            f = atoms.get_forces()
            if f is not None:
                forces = np.asarray(f)
        except Exception:
            pass

        return ParsedStructure(
            atoms=atoms,
            source_format="vasp",
            source_path=path,
            total_energy=energy,
            forces=forces,
        )

    @staticmethod
    def parse_lammps_dump(path: Path) -> ParsedStructure:
        """Parse the last frame from a LAMMPS dump file using ASE.

        Parameters
        ----------
        path:
            Path to the LAMMPS dump file.

        Returns
        -------
        ParsedStructure

        Raises
        ------
        ImportError
            If ``ase`` is not installed.
        """
        try:
            import ase.io
        except ImportError as exc:
            raise ImportError(
                "ase is required for LAMMPS parsing. Install with: pip install ase"
            ) from exc

        atoms = ase.io.read(str(path), index=-1, format="lammps-dump-text")

        forces: np.ndarray | None = None
        try:
            f = atoms.get_forces()
            if f is not None:
                forces = np.asarray(f)
        except Exception:
            pass

        return ParsedStructure(
            atoms=atoms,
            source_format="lammps",
            source_path=path,
            total_energy=None,
            forces=forces,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, structure: ParsedStructure, output_dir: Path) -> Path:
        """Render a three-panel orthogonal-projection PNG of the structure.

        The three panels show views along the z-, x-, and y-axes respectively.
        A text annotation at the bottom lists key ground-truth parameters.

        Parameters
        ----------
        structure:
            Parsed structure to visualise.
        output_dir:
            Directory where the PNG will be saved.

        Returns
        -------
        Path
            Resolved absolute path to the saved PNG.

        Raises
        ------
        ImportError
            If ``matplotlib`` or ``ase`` are not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from ase.visualize.plot import plot_atoms
        except ImportError as exc:
            raise ImportError(
                "matplotlib and ase are required for rendering. "
                "Install with: pip install matplotlib ase"
            ) from exc

        out_path = output_dir / f"{structure.source_path.stem}.png"
        rotations = [
            ("0x,0y,0z", "View ‖ z"),
            ("90x,0y,0z", "View ‖ x"),
            ("0x,90y,0z", "View ‖ y"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"{structure.chemical_formula}  ({structure.n_atoms} atoms)",
            fontsize=13,
            fontweight="bold",
        )

        for ax, (rot, label) in zip(axes, rotations):
            plot_atoms(structure.atoms, ax, radii=0.4, rotation=rot)
            ax.set_title(label, fontsize=9)
            ax.axis("off")

        # Annotation block with GT parameters
        a, b, c = structure.lattice_constants
        lines = [
            f"Formula: {structure.chemical_formula}   |   Atoms: {structure.n_atoms}",
            f"Elements: {', '.join(structure.unique_elements)}",
            f"a = {a:.4f} Å   b = {b:.4f} Å   c = {c:.4f} Å",
            f"Vol = {structure.volume:.3f} Å³   ρ = {structure.density:.4f} g/cm³",
        ]
        if structure.total_energy is not None:
            lines.append(
                f"E_tot = {structure.total_energy:.4f} eV   "
                f"E/atom = {structure.total_energy / structure.n_atoms:.4f} eV/atom"
            )
        if structure.mean_force_magnitude is not None:
            lines.append(
                f"|F|_avg = {structure.mean_force_magnitude:.4f} eV/Å   "
                f"|F|_max = {structure.max_force_magnitude:.4f} eV/Å"
            )

        fig.text(
            0.5,
            -0.01,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path.resolve()

    # ------------------------------------------------------------------
    # QA generation
    # ------------------------------------------------------------------

    def _generate_qa_pairs(
        self,
        structure: ParsedStructure,
        n_per_difficulty: int = 2,
    ) -> list[QAPair]:
        """Sample QA pairs at each difficulty level from the template pool."""
        all_pairs: list[QAPair] = []
        pool = _build_template_pool(structure)

        for diff in ("easy", "medium", "hard"):
            tier = [(q, fn) for q, fn, d in pool if d == diff]
            self._rng.shuffle(tier)
            for question, ans_fn in tier[:n_per_difficulty]:
                try:
                    answer = ans_fn(structure)
                    all_pairs.append(QAPair(question=question, answer=answer, difficulty=diff))
                except Exception:
                    continue

        return all_pairs
