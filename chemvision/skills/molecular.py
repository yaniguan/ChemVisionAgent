"""Skill: extract structural information from molecular diagrams."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.base import BaseSkill
from chemvision.skills.outputs import FunctionalGroup, MolecularStructureData, StereocenterInfo

_PROMPT_TEMPLATE = """\
You are an expert organic chemist with deep knowledge of structural analysis.

Examine this molecular structure diagram — it may be a 2-D skeletal formula,
a 3-D ball-and-stick model, a space-filling model, or a hand-drawn structure —
and extract ALL of the following:

1. SMILES string (canonical, if determinable from the image)
2. IUPAC name (if determinable)
3. Common or trade name (if labelled)
4. Molecular formula (count all heavy atoms visible)
5. Estimated molecular weight (g/mol)
6. All functional groups present (with occurrence counts)
7. All stereocenters (R/S) and geometric isomerism (E/Z) elements
8. Ring systems present (e.g. benzene, cyclohexane, pyridine, piperidine)

Your output MUST be a single valid JSON object:
{{
  "smiles": "<SMILES string or null>",
  "iupac_name": "<IUPAC name or null>",
  "common_name": "<common/trade name or null>",
  "molecular_formula": "<e.g. C6H12O6 or null>",
  "molecular_weight": <number in g/mol or null>,
  "functional_groups": [
    {{
      "name": "<e.g. hydroxyl | carboxyl | amine | carbonyl | ester | amide | ether | halide | nitro | thiol | phosphate | alkene | alkyne | aromatic>",
      "smarts": "<SMARTS pattern or null>",
      "count": <integer>
    }}
  ],
  "stereocenters": [
    {{
      "atom_or_bond": "<e.g. C3 or C2=C3>",
      "descriptor": "<R | S | E | Z>",
      "confidence": <0.0-1.0>
    }}
  ],
  "ring_systems": ["<e.g. benzene>", "<e.g. cyclohexane>"],
  "num_rings": <integer or null>,
  "confidence": <0.0-1.0>
}}

Guidelines:
- If the image contains multiple separate molecules, analyse the largest / most prominent one.
- Use null for fields that cannot be determined from the image.
- For SMILES, prefer canonical form; use stereochemistry notation (@, @@, /, \\\\) when stereocenters are visible.
- Respond with only the JSON object, no other text.
"""


class MolecularStructureSkill(BaseSkill):
    """Extract SMILES, functional groups, stereocenters, and ring systems
    from 2-D or 3-D molecular structure diagrams.

    Handles skeletal formulae, ball-and-stick models, space-filling models,
    and hand-drawn structures. Returns a typed :class:`MolecularStructureData`
    with full structural annotation.

    Example
    -------
    >>> skill = MolecularStructureSkill()
    >>> result = skill(image, model)
    >>> result.smiles
    'CC(=O)Oc1ccccc1C(=O)O'
    >>> result.iupac_name
    '2-(acetyloxy)benzoic acid'
    >>> [g.name for g in result.functional_groups]
    ['ester', 'carboxyl', 'aromatic']
    >>> result.num_rings
    1
    """

    name = "molecular_structure"

    def build_prompt(self, **kwargs: Any) -> str:
        return _PROMPT_TEMPLATE

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> MolecularStructureData:
        """Run molecular structure extraction and return a typed
        :class:`MolecularStructureData`.

        Parameters
        ----------
        image:
            RGB PIL image of the molecular structure diagram.
        model:
            Loaded vision model (local or
            :class:`~chemvision.agent.adapter.AnthropicVisionFallback`).
        """
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        functional_groups: list[FunctionalGroup] = []
        for fg in to_list(data.get("functional_groups")):
            if not isinstance(fg, dict):
                continue
            count_raw = fg.get("count")
            functional_groups.append(
                FunctionalGroup(
                    raw_output=raw,
                    name=to_str(fg.get("name"), ""),
                    smarts=to_str(fg.get("smarts")) or None,
                    count=int(count_raw) if count_raw is not None else 1,
                )
            )

        stereocenters: list[StereocenterInfo] = []
        for sc in to_list(data.get("stereocenters")):
            if not isinstance(sc, dict):
                continue
            stereocenters.append(
                StereocenterInfo(
                    raw_output=raw,
                    atom_or_bond=to_str(sc.get("atom_or_bond"), ""),
                    descriptor=to_str(sc.get("descriptor"), ""),
                    confidence=to_float(sc.get("confidence")),
                )
            )

        ring_systems = [to_str(r) for r in to_list(data.get("ring_systems")) if r]

        num_rings_raw = data.get("num_rings")
        num_rings = int(num_rings_raw) if num_rings_raw is not None else None

        return MolecularStructureData(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            smiles=to_str(data.get("smiles")) or None,
            iupac_name=to_str(data.get("iupac_name")) or None,
            common_name=to_str(data.get("common_name")) or None,
            molecular_formula=to_str(data.get("molecular_formula")) or None,
            molecular_weight=to_float(data.get("molecular_weight")),
            functional_groups=functional_groups,
            stereocenters=stereocenters,
            ring_systems=ring_systems,
            num_rings=num_rings,
        )
