# ChemVision Agent — Project Memory

## Structure
Scaffold complete. Key layout:
- `chemvision/data/` — DatasetBuilder, ImageRecord, DatasetConfig
- `chemvision/models/` — BaseVisionModel, QwenVLWrapper, PeftFineTuner, ModelConfig/PeftConfig
- `chemvision/audit/` — AuditRunner, AuditReport, AuditConfig
- `chemvision/skills/` — BaseSkill + registry + SpectrumReadingSkill / MolecularStructureSkill / MicroscopySkill
- `chemvision/agent/` — ChemVisionAgent (ReAct), AgentTrace, AgentConfig
- `chemvision/api.py` — FastAPI app (serve, reason, audit endpoints)
- `chemvision/cli.py` — Typer CLI (serve, reason, audit commands)

## Tooling
- uv for dependency management; `.venv` at project root
- `uv run pytest` to execute tests
- `[tool.hatch.build.targets.wheel] packages = ["chemvision"]` required in pyproject.toml

## Data pipeline (fully implemented)
- `synthetic.py` — VASP OUTCAR + LAMMPS dump → ASE Atoms → matplotlib render → QA pairs
  - `SyntheticGenerator.generate(file_path, output_dir)` is the entry point
  - `ParsedStructure` wraps ase.Atoms with derived properties (density, lattice, forces)
  - `classify_bravais()` is a module-level helper used in tests
  - Heavy deps (ase, matplotlib) imported lazily inside methods
- `scraper.py` — DOI/arXiv → PDF (requests) → figures (pypdf) → QA (Claude Haiku claude-haiku-4-5)
  - `LiteratureScraper.scrape(identifiers)` is the entry point
  - arXiv uses direct arxiv.org/pdf URL; DOIs use Unpaywall
  - `parse_captions()` is a static method tested independently
- `builder.py` — fully implemented: collects images, splits, saves HuggingFace DatasetDict
  - `DatasetBuilder.save()` uses `model_dump(mode='json')` for Path→str coercion
- `data_stats.py` — standalone Rich script; also has `print_stats(path)` callable
- `schema.py` extended: `ImageRecord` has `bbox`, `difficulty`, `source`; new domains `CRYSTAL_STRUCTURE`, `SIMULATION`; new `SyntheticConfig`, `ScraperConfig`

## Status
47/47 tests pass. Next steps: implement QwenVLWrapper, then wire agent skills.
