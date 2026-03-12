# ChemVision Agent

A multimodal scientific image reasoning platform built on vision-language models.

## Architecture

```
chemvision/
  data/      # Dataset construction pipeline (collect → annotate → split)
  models/    # Vision model wrappers (QwenVL) and LoRA fine-tuning
  audit/     # Capability auditing framework with per-skill benchmarks
  skills/    # Composable vision skill modules (spectrum, molecular, microscopy)
  agent/     # ReAct-based orchestration loop
  api.py     # FastAPI REST service
  cli.py     # Typer CLI entry point
```

## How to run

### Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### CLI

```bash
# Start the API server
chemvision serve --port 8000

# Run the agent on an image
chemvision reason path/to/spectrum.png "What solvent was used?"

# Audit model capabilities
chemvision audit path/to/image.png --skill spectrum_reading --skill microscopy
```

### API server

```bash
chemvision serve --reload
# → http://localhost:8000/docs
```

### Tests

```bash
pytest
```
