"""Skill: extract reaction type, molecules, and experimental conditions
from a chemistry literature figure (reaction scheme, table, or diagram)."""

from __future__ import annotations

from typing import Any

from PIL.Image import Image

from chemvision.skills._parse import extract_json, to_float, to_list, to_str
from chemvision.skills.base import BaseSkill, SkillResult
from chemvision.skills.outputs import Molecule, ReactionConditions, ReactionData

_PROMPT_TEMPLATE = """\
You are an expert synthetic chemist reading a chemistry literature figure.

Analyse this image — it may be a reaction scheme, a conditions table, or a \
multi-step synthesis diagram — and extract ALL of the following:

1. Reaction type (e.g. Suzuki coupling, aldol condensation, Diels-Alder, \
   ring-opening polymerisation, photocatalytic oxidation …)
2. Every molecule visible: reactants, reagents, catalysts, solvents, products
3. Experimental conditions shown (temperature, pressure, solvent, time, atmosphere, yield)
4. Arrow type (single-step, multi-step, retrosynthetic, equilibrium)

Your output MUST be a single valid JSON object:
{{
  "reaction_type": "<reaction name or 'unknown'>",
  "arrow_type": "<single-step | multi-step | retrosynthetic | equilibrium | unknown>",
  "molecules": [
    {{
      "name": "<IUPAC or common name>",
      "smiles": "<SMILES string or null>",
      "role": "<reactant | product | reagent | catalyst | solvent | unknown>"
    }}
  ],
  "conditions": {{
    "temperature": "<value with unit, e.g. 80 °C, rt, -78 °C, or null>",
    "pressure": "<value with unit, e.g. 1 atm, or null>",
    "solvent": "<solvent name(s) or null>",
    "time": "<duration with unit, e.g. 12 h, or null>",
    "atmosphere": "<N2 | Ar | air | O2 | H2 | or null>",
    "yield_percent": <number 0-100 or null>
  }},
  "confidence": <0.0-1.0>
}}

If the image does not contain a reaction scheme, still return the JSON with \
empty/null fields and a low confidence score.
Respond with only the JSON object, no other text.\
"""


class ExtractReactionSkill(BaseSkill):
    """Extract reaction type, molecules, and experimental conditions from
    a chemistry literature figure.

    Handles reaction schemes, multi-step syntheses, conditions tables, and
    retrosynthetic analyses.  Returns a typed :class:`ReactionData` with
    molecule roles (reactant/product/catalyst/…) and all observed
    experimental parameters.

    Example
    -------
    >>> skill = ExtractReactionSkill()
    >>> result = skill(image, model)
    >>> result.reaction_type
    'Suzuki coupling'
    >>> result.conditions.temperature
    '80 °C'
    >>> [m.name for m in result.molecules if m.role == 'product']
    ['4-phenylacetophenone']
    """

    name = "extract_reaction"

    def build_prompt(self, **kwargs: Any) -> str:
        return _PROMPT_TEMPLATE

    def __call__(self, image: Image, model: Any, **kwargs: Any) -> ReactionData:
        prompt = self.build_prompt(**kwargs)
        raw = model.generate(image, prompt)
        data = extract_json(raw) or {}

        molecules: list[Molecule] = []
        for m in to_list(data.get("molecules")):
            if isinstance(m, dict):
                role = to_str(m.get("role"), "unknown")
                if role not in {"reactant", "product", "reagent", "catalyst", "solvent", "unknown"}:
                    role = "unknown"
                molecules.append(
                    Molecule(
                        name=to_str(m.get("name"), ""),
                        smiles=to_str(m.get("smiles")) or None,
                        role=role,  # validated above
                        raw_output=raw,
                    )
                )

        cond_raw = data.get("conditions") or {}
        conditions: ReactionConditions | None = None
        if isinstance(cond_raw, dict):
            conditions = ReactionConditions(
                temperature=to_str(cond_raw.get("temperature")) or None,
                pressure=to_str(cond_raw.get("pressure")) or None,
                solvent=to_str(cond_raw.get("solvent")) or None,
                time=to_str(cond_raw.get("time")) or None,
                atmosphere=to_str(cond_raw.get("atmosphere")) or None,
                yield_percent=to_float(cond_raw.get("yield_percent")),
                raw_output=raw,
            )

        return ReactionData(
            skill_name=self.name,
            raw_output=raw,
            parsed=data,
            confidence=to_float(data.get("confidence")),
            reaction_type=to_str(data.get("reaction_type"), ""),
            arrow_type=to_str(data.get("arrow_type"), ""),
            molecules=molecules,
            conditions=conditions,
        )
