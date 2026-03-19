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


 Getting Started                                                                              
                                                                                               
  1. Install dependencies                                                                      
                                                                                               
  cd ChemVisionAgent                                                                           
  pip install uv          # if not already installed                                           
  uv sync                 # installs everything from uv.lock                                   
  uv add streamlit pandas # for the demo app (not in lock file yet)
                                                                                               
  2. Set your API key

  export ANTHROPIC_API_KEY=sk-ant-...                                                          
                                                                                               
  That's the only required credential. No GPU, no model download — the default backend uses    
  Claude as the vision model.                                                                  
                                                                                               
  ---                                                                                          
  Three ways to run it                                                                         
                                                                                               
  A. Streamlit demo (easiest)                                                                  

  uv run streamlit run demo/app.py
  # → http://localhost:8501
  Upload 1–N images, type a question, watch the reasoning trace live.

  B. FastAPI server

  uv run uvicorn chemvision.api:app --reload
  # → http://localhost:8000/docs  (Swagger UI)

  curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{
      "question": "What crystal phase dominates this XRD pattern?",
      "image_paths": ["/path/to/xrd.png"]
    }'

  C. Python API (embedding in your own code)

  from chemvision.agent import ChemVisionAgent, AgentConfig

  agent = ChemVisionAgent(AgentConfig(
      anthropic_api_key="sk-ant-...",
      confidence_threshold=0.75,   # flag conclusions below this
      max_steps=10,
      verbose=True,                # stream Thought/Action/Observation
  ))

  report = agent.run(
      question="Identify all anomalies and their severity.",
      image_paths=["sem_coating.png"],
  )

  print(report.final_answer)
  print(report.structured_data)         # parsed JSON fields
  print(report.low_confidence_flag)     # True if any skill < 0.75
  for log in report.tool_logs:
      print(log.skill_name, log.confidence)



Main features & their entry points

  Vision skills (use standalone or via agent)

  from chemvision.skills import (
      AnalyzeStructureSkill,
      ExtractSpectrumSkill,
      CompareStructuresSkill,
      ValidateCaptionSkill,
      DetectAnomalySkill,
  )
  from chemvision.agent.adapter import AnthropicVisionFallback
  from PIL import Image

  model = AnthropicVisionFallback()          # or LLaVAWrapper for local GPU
  image = Image.open("xrd.png")

  peaks = ExtractSpectrumSkill()(image, model, spectrum_type="XRD")
  print(peaks.peaks)                         # list[Peak] with two_theta, intensity

  report = DetectAnomalySkill()(
      image, model, domain_context="SEM image of TBC coating"
  )
  print(report.severity, report.anomalies)

  Capability audit

  python -m chemvision.audit.run \
    --model claude-sonnet-4-20250514 \
    --dataset data/benchmark/ \
    --output reports/
  # → reports/audit_report.md  with embedded heatmap

  from chemvision.audit import CapabilityMatrix, DegradationTester

  matrix = CapabilityMatrix()
  matrix.run_evaluation(model, dataset)
  matrix.export_heatmap("heatmap.png")      # 5 tasks × 3 difficulties

  tester = DegradationTester()
  envelope = tester.test_all(model, dataset)
  envelope.save_json("reliability.json")    # critical noise/blur/JPEG thresholds

  Synthetic XRD data generation

  from chemvision.data.synthetic_generator import XRDImageGenerator

  gen = XRDImageGenerator(transition_temp=500.0, seed=42)
  samples = gen.generate_temperature_series(
      temperatures=[200, 300, 400, 500, 600, 700],
      output_dir="data/xrd_series/",
  )
  for s in samples:
      print(f"{s.temperature_c}°C — {s.dominant_phase}, D={s.grain_size_nm:.1f} nm")

  Register a custom skill

  from chemvision.skills import register_skill, BaseSkill, SkillResult
  from PIL import Image

  class GrainCountSkill(BaseSkill):
      name = "grain_count"

      def __call__(self, image: Image, model, **kwargs) -> SkillResult:
          prompt = "Count the number of distinct grains. Return JSON: {count, confidence}"
          raw = model.generate(image, prompt)
          data = extract_json(raw) or {}
          return SkillResult(
              skill_name=self.name,
              raw_output=raw,
              parsed=data,
              confidence=data.get("confidence"),
          )

  register_skill(GrainCountSkill())
  # Now any agent with skill_names=["grain_count"] can call it

  Test suite

  uv run pytest                       # 265 tests, ~2s
  uv run pytest tests/test_e2e.py -v  # e2e scenario only



  Improvement roadmap

  Fine-tuning

  What exists: PeftFineTuner + configs/lora_train.yaml + scripts/train_lora.py are already
  scaffolded. The training loop, collate function, and LoRA config are implemented.

  What to do next:

  1. Build a labeled dataset using LiteratureScraper to pull XRD/SEM figures + captions from
  arXiv papers, then add ground-truth skill outputs as annotations.
  2. Run LoRA fine-tuning on LLaVA-1.6:
  python scripts/train_lora.py configs/lora_train.yaml
  # Checkpoints → checkpoints/lora-v1/
  3. Domain-adaptive pretraining: Before LoRA, do continued pretraining on materials science
  text (MatBERT-style) to improve the base language model's domain vocabulary before vision
  fine-tuning.
  4. Task-specific heads: Add a regression head for direct lattice parameter prediction instead
   of relying on JSON parsing from free text.

  ---
  RLHF (Reinforcement Learning from Human Feedback)

  The current agent uses confidence scores as a proxy for quality. To do real RLHF:

  Step 1 — Collect preferences. Add a feedback endpoint to the API:
  # POST /feedback  body: {report_id, preferred_answer, rejected_answer, reason}
  Store (question, image, chosen_response, rejected_response) triplets.

  Step 2 — Train a reward model. Fine-tune a smaller vision-language model (e.g., LLaVA-7B) on
  the preference pairs using Bradley-Terry loss. The reward model learns to score (image,
  question, answer) → scalar quality score.

  Step 3 — PPO/DPO fine-tuning. Use the reward model to optimize the policy model with PPO (via
   trl library) or the simpler Direct Preference Optimization (DPO), which doesn't need a
  separate RM:
  pip install trl
  # Use trl.DPOTrainer with your preference dataset

  Step 4 — Iterate. The Streamlit demo is the ideal human feedback interface — add thumbs
  up/down buttons next to each tool call observation and the final answer.

  ---
  Alignment (safety + accuracy)

  Several alignment problems are specific to scientific AI:

  Hallucination prevention:
  - The ValidateCaptionSkill already flags image/text inconsistency — extend this into a
  post-generation verifier that checks whether the final answer is supported by the skill
  observations.
  - Add a "citation" step to the ReAct loop: before calling final_answer, force the agent to
  cite which specific skill observation supports each claim.

  Confidence calibration:
  - Current confidence scores are self-reported by the vision model (not calibrated). Add a
  calibration step: compare self-reported confidence against ground-truth accuracy on the audit
   benchmark, then fit an isotonic regression or Platt scaling transform.
  - The DegradationTester.reliability_envelope gives you the degradation-robust confidence
  regions for free.

  Uncertainty quantification:
  - Run each skill k=5 times with temperature>0, compute ensemble variance as an uncertainty
  estimate instead of trusting a single self-reported confidence.

  ---
  Multi-modal alignment

  This is the most research-forward direction and directly relevant to materials science:

  Cross-modal grounding:
  - The ChainOfVisionReasoning forces LOCALIZE → ANALYZE → CONCLUDE structure. Extend LOCALIZE
  to output bounding boxes, then verify that the ANALYZE step's claims actually refer to
  regions inside those boxes. This is visual grounding / region-level alignment.

  Structure-spectrum alignment:
  - A key materials science task: given an XRD spectrum and a TEM image of the same sample, the
   agent should align peak assignments in the spectrum to crystal planes visible in the TEM.
  This is genuinely multi-modal within a single query — add a new skill
  align_spectrum_to_structure(xrd_image, tem_image).

  Multi-scale alignment:
  - Samples are often characterized at multiple scales (macro SEM → micro TEM → atomic STEM).
  Build a MultiScaleAgentConfig that assigns different skills to different scales and then
  synthesizes across them, similar to how compare_structures works but with a hierarchy.

  Text-structure alignment (ChemDraw / SMILES):
  - Extend MolecularStructureSkill to output a SMILES string, then validate it against a
  chemistry toolkit (RDKit) to ensure the extracted structure is chemically valid. This catches
   hallucinated bonds.

  Physics-constrained decoding:
  - The biggest alignment gap in materials science AI: generated lattice parameters should
  satisfy space group symmetry constraints. Add a post-processing step that projects the
  model's output lattice parameters onto the nearest physically valid point for the predicted
  space group (using spglib).

  ---
  Practical priority order

  ┌──────────┬─────────────────────────────────────────────────┬───────────────────────────┐
  │ Priority │                     Action                      │          Impact           │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 1        │ Build a 500-sample labeled dataset with         │ Unlocks everything else   │
  │          │ LiteratureScraper                               │                           │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 2        │ Run LoRA fine-tuning with the existing script   │ +15–25% skill accuracy    │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 3        │ Add calibration layer on top of confidence      │ Trustworthy confidence    │
  │          │ scores                                          │ flags                     │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 4        │ Add thumbs up/down in Streamlit → preference    │ Foundation for RLHF       │
  │          │ dataset                                         │                           │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 5        │ DPO fine-tuning on preference pairs             │ Reduces hallucination     │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 6        │ Structure-spectrum cross-modal alignment skill  │ Novel research            │
  │          │                                                 │ contribution              │
  ├──────────┼─────────────────────────────────────────────────┼───────────────────────────┤
  │ 7        │ Physics-constrained decoding (spglib)           │ Production-ready for labs │
  └──────────┴─────────────────────────────────────────────────┴───────────────────────────┘