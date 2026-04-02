# ChemVisionAgent — Full System Pipeline Specification

> Owner: yaniguan | Version: 0.3.0 | Updated: 2026-04-01

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Pipeline Stage Reference](#2-pipeline-stage-reference)
   - Stage 0: Input Ingestion
   - Stage 1: Vision Perception (VLM)
   - Stage 2: Retrieval Grounding (RAG)
   - Stage 3: Molecular Encoding
   - Stage 4: Physics Constraints
   - Stage 5: Property Prediction
   - Stage 6: Generative Optimisation (Pareto MCTS)
   - Stage 7: Agent Orchestration (ReAct)
   - Stage 8: Audit & Evaluation
3. [Data Schema Reference](#3-data-schema-reference)
4. [Model Backend Reference](#4-model-backend-reference)
5. [Deployment Architecture](#5-deployment-architecture)
6. [Dependency Map](#6-dependency-map)
7. [Performance Characteristics](#7-performance-characteristics)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            DELIVERY SURFACES                                    │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────────────┐    │
│   │ Streamlit │    │ FastAPI REST │    │ Typer CLI│    │ Python Library    │    │
│   │ demo/app  │    │ /analyze     │    │ chemvision│   │ agent.run()       │    │
│   └─────┬─────┘    └──────┬───────┘    └─────┬────┘    └────────┬──────────┘    │
│         └──────────────────┴─────────────────┴──────────────────┘              │
│                                    │                                            │
├────────────────────────────────────┼────────────────────────────────────────────┤
│                           ORCHESTRATION LAYER                                   │
│                                    │                                            │
│   ┌────────────────────────────────▼───────────────────────────────────────┐    │
│   │               ChemVisionAgent (ReAct Loop)                             │    │
│   │                                                                        │    │
│   │  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────────┐  │    │
│   │  │ Thought  │──▶│   Action    │──▶│ Observation  │──▶│ Loop / End │  │    │
│   │  │ (Claude) │   │ (Skill Call)│   │ (Skill Out) │   │            │  │    │
│   │  └──────────┘   └──────┬──────┘   └──────────────┘   └────────────┘  │    │
│   │                        │                                               │    │
│   │         AgentPlanner (Claude API + Tool-Use Protocol)                  │    │
│   └────────────────────────┼───────────────────────────────────────────────┘    │
│                            │                                                    │
├────────────────────────────┼────────────────────────────────────────────────────┤
│                      SKILL REGISTRY (9 Skills)                                  │
│                            │                                                    │
│   ┌────────────┬───────────┼───────────┬───────────────┬──────────────────┐    │
│   │            │           │           │               │                  │    │
│   ▼            ▼           ▼           ▼               ▼                  ▼    │
│ ┌──────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────┐  ┌──────────┐  │
│ │Struct│  │Spectrum│  │Compare │  │Caption │  │ Anomaly    │  │Reaction  │  │
│ │ure   │  │Extract │  │Struct. │  │Validate│  │ Detection  │  │Extract   │  │
│ └──┬───┘  └───┬────┘  └───┬────┘  └───┬────┘  └─────┬──────┘  └────┬─────┘  │
│    │          │            │           │             │              │        │
│ ┌──┴───┐  ┌──┴────┐  ┌───┴────┐  ┌───┴────┐  ┌────┴──────┐  ┌──┴──────┐  │
│ │Micro │  │Molecu │  │Property│  │        │  │           │  │         │  │
│ │scopy │  │lar    │  │Predict.│  │        │  │           │  │         │  │
│ └──────┘  └───────┘  └───┬────┘  └────────┘  └───────────┘  └─────────┘  │
│                           │                                                │
├───────────────────────────┼────────────────────────────────────────────────┤
│                    SCIENTIFIC PIPELINE                                      │
│                           │                                                │
│   ┌───────────────────────▼────────────────────────────────────────────┐  │
│   │                PropertyPredictionSkill Chain                       │  │
│   │                                                                    │  │
│   │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌───────┐ │  │
│   │  │ PubChem │  │ Morgan   │  │ RDKit    │  │ Pareto │  │Vector │ │  │
│   │  │ Retriev.│  │ Encoder  │  │ Predict. │  │ MCTS   │  │ Store │ │  │
│   │  └────┬────┘  └────┬─────┘  └────┬─────┘  └───┬────┘  └───┬───┘ │  │
│   │       │             │             │            │            │     │  │
│   └───────┼─────────────┼─────────────┼────────────┼────────────┼─────┘  │
│           │             │             │            │            │        │
├───────────┼─────────────┼─────────────┼────────────┼────────────┼────────┤
│       FOUNDATION LAYERS │             │            │            │        │
│           │             │             │            │            │        │
│   ┌───────▼──────┐ ┌───▼───────┐ ┌───▼──────┐ ┌──▼────┐ ┌───▼──────┐ │
│   │   PubChem    │ │   RDKit   │ │  spglib  │ │ MACE  │ │ ChromaDB │ │
│   │  REST API    │ │ Morgan FP │ │ Symmetry │ │ (opt) │ │ (opt)    │ │
│   │ (no key)     │ │ ETKDG 3D  │ │ Scherrer │ │       │ │          │ │
│   └──────────────┘ └───────────┘ └──────────┘ └───────┘ └──────────┘ │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│                      VISION BACKENDS                                    │
│                                                                        │
│   ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│   │ Anthropic Claude │  │ LLaVA-1.6 (7B)  │  │  InternVL2 (8B)  │   │
│   │ (Cloud, default) │  │ (Local GPU)      │  │  (Local GPU)     │   │
│   └──────────────────┘  └──────────────────┘  └───────────────────┘   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Stage Reference

### Stage 0: Input Ingestion

```
PURPOSE:  Accept user query + scientific images from any delivery surface
LATENCY:  < 50 ms
```

| Field | Spec |
|---|---|
| **Input formats** | PNG, JPG, TIFF, BMP (any PIL-supported raster) |
| **Max image size** | Resized to fit Claude's vision limit (5 MB / 8192 px longest edge) |
| **Encoding** | Raw bytes → PIL.Image.open() → RGB → base64 PNG for Claude API |
| **Text input** | Free-form natural language question (UTF-8, max ~2000 tokens) |
| **Multi-image** | Up to 20 images per query (agent resolves via `image_index` kwargs) |

**Data flow:**
```
User input
  ├── question: str                          "What crystal phase is shown in this XRD?"
  └── image_paths: list[str]                 ["/data/xrd_500C.png", "/data/xrd_700C.png"]
          │
          ▼
  PIL.Image.open(path).convert("RGB")       → list[Image.Image]
          │
          ▼
  base64.b64encode(BytesIO → PNG bytes)     → list[str]  (for Claude API content blocks)
```

**Code path:** `ChemVisionAgent.run()` → `AgentPlanner.build_initial_message()`

---

### Stage 1: Vision Perception (VLM Skill Execution)

```
PURPOSE:  Extract structured scientific data from images via prompt-driven VLM
LATENCY:  1–8 s per skill call (cloud) | 3–15 s (local GPU)
```

**Available backends (mutually exclusive per session):**

| Backend | Model | Device | Params | Input | Output |
|---|---|---|---|---|---|
| **AnthropicVisionFallback** | claude-sonnet-4-20250514 | Cloud API | ~70B+ | PIL.Image + str prompt | str (raw text / JSON) |
| **LLaVAWrapper** | llava-v1.6-mistral-7b-hf | CUDA GPU | 7.6B | PIL.Image + str prompt | str |
| **LLaVAWrapper** | InternVL2-8B | CUDA GPU | 8B | PIL.Image + str prompt | str |

**Backend selection logic:**
```python
if config.model is not None:
    # Local GPU model (LLaVA / InternVL2)
    model = LLaVAWrapper(config.model)
    model.load()
else:
    # Cloud fallback (Anthropic Claude)
    model = AnthropicVisionFallback(
        api_key=config.anthropic_api_key,
        model=config.planning_model,      # claude-sonnet-4-20250514
        max_tokens=4096,
    )
```

**Skill execution per skill type:**

| # | Skill | Registry Name | Input | Output Model | Key Fields |
|---|---|---|---|---|---|
| 1 | AnalyzeStructure | `analyze_structure` | Image + `material_type: str` | `StructureAnalysis` | lattice_params(a,b,c,α,β,γ), symmetry, defect_locations[{x,y,type,confidence}], defect_density |
| 2 | ExtractSpectrum | `extract_spectrum_data` | Image + `spectrum_type: "XRD"\|"Raman"\|"XPS"` | `SpectrumData` | peaks[{position,intensity,assignment,fwhm}], snr, background_level |
| 3 | CompareStructures | `compare_structures` | list[Image] + `comparison_type: str` | `StructureComparison` | diff_regions[{x,y,w,h,desc}], quantitative_changes[{metric,before,after,delta}], trend |
| 4 | ValidateCaption | `validate_figure_caption` | Image + `caption: str` | `CaptionValidation` | consistency_score(0–1), contradictions[str] |
| 5 | DetectAnomaly | `detect_anomaly` | Image + `domain_context: str` | `AnomalyReport` | anomalies[{x,y,type,severity,confidence}], severity, recommendations[str] |
| 6 | ExtractReaction | `extract_reaction` | Image | `ReactionData` | reaction_type, molecules[{name,smiles,role}], conditions{temp,pressure,solvent,time,yield}, arrow_type |
| 7 | Microscopy | `analyze_microscopy` | Image + `imaging_context: str` | `MicroscopyAnalysis` | morphology, particles[{diameter,shape,aspect_ratio}], size_statistics, scale_bar, modality, magnification |
| 8 | MolecularStructure | `molecular_structure` | Image | `MolecularStructureData` | smiles, iupac_name, formula, mw, functional_groups, stereocenters, ring_systems |
| 9 | PropertyPrediction | `property_prediction` | Image (or `smiles: str` kwarg) + `n_mcts_iterations: int`, `run_optimisation: bool` | `PropertyPredictionResult` | input_smiles, pubchem_*, predicted(PropertyResult), similar_molecules, pareto_candidates |

**Internal data flow per skill call:**
```
model.generate(image, prompt) → raw_text: str
                                    │
                                    ▼
                        extract_json(raw_text) → dict | None
                                    │
                                    ▼
                        Parse into typed Pydantic model
                        with confidence score [0.0, 1.0]
                                    │
                                    ▼
                        SkillResult (skill_name, raw_output,
                                     parsed: dict, confidence: float)
```

---

### Stage 2: Retrieval Grounding (RAG)

```
PURPOSE:  Ground VLM outputs in verified chemical databases to prevent hallucination
LATENCY:  200–800 ms (PubChem REST) | < 1 ms (vector store)
```

#### 2a. PubChem REST Client

| Field | Spec |
|---|---|
| **Module** | `chemvision.retrieval.pubchem_client.PubChemClient` |
| **API** | PubChem PUG-REST (`https://pubchem.ncbi.nlm.nih.gov/rest/pug/`) |
| **Auth** | None required (public API) |
| **Rate limit** | 5 req/s (enforced by 0.1 s sleep between similarity calls) |
| **Cache** | `functools.lru_cache(256)` per session |
| **Timeout** | 15 s default |

**Methods & I/O:**

```
fetch_by_smiles(smiles: str) → dict[str, Any]
fetch_by_name(name: str)     → dict[str, Any]
fetch_by_cid(cid: int)       → dict[str, Any]
get_similar_compounds(smiles, threshold=90, max_results=10) → list[dict]
```

**Property fields returned:**
```python
{
    "CID": 2244,
    "MolecularFormula": "C9H8O4",
    "MolecularWeight": "180.16",
    "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "IsomericSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "IUPACName": "2-acetyloxybenzoic acid",
    "XLogP": 1.2,
    "TPSA": 63.6,
    "HBondDonorCount": 1,
    "HBondAcceptorCount": 4,
    "RotatableBondCount": 3,
    "HeavyAtomCount": 13,
    "Complexity": 212.0,
}
```

**Error handling:** All methods return `{}` on network/parse failure — callers never raise.

#### 2b. Molecule Vector Store

| Field | Spec |
|---|---|
| **Module** | `chemvision.retrieval.vector_store.MoleculeVectorStore` |
| **Embedding dim** | 2048 (Morgan ECFP4) or 512 (Uni-Mol2 when available) |
| **Distance metric** | Cosine similarity (embeddings L2-normalised on insert) |
| **Backend (default)** | Pure numpy: `matrix @ query` → argsort → top-k |
| **Backend (persistent)** | ChromaDB with `hnsw:space=cosine` |
| **Capacity** | In-memory: ~100K molecules on 16 GB RAM |

**Methods & I/O:**
```
add(name: str, embedding: np.ndarray, metadata: dict) → None
add_batch(names, embeddings, metadatas) → None
search(query: np.ndarray, k=5) → list[{"name": str, "score": float, **metadata}]
save(path) → writes .npz + .json
load(path) → restores from disk
```

**Data flow in PropertyPredictionSkill:**
```
SMILES
  │
  ▼
MolecularEncoder.encode(smiles) → np.ndarray (2048,)
  │
  ├──▶ store.add(smiles, embedding, metadata)     ← index this molecule
  │
  └──▶ store.search(embedding, k=5)               ← find similar known molecules
            │
            ▼
       list[{"name": "ibuprofen", "score": 0.87, "smiles": "CC(C)Cc1ccc(cc1)..."}]
```

---

### Stage 3: Molecular Encoding

```
PURPOSE:  Convert SMILES to numerical representations (fingerprints, 3D conformers, descriptors)
LATENCY:  10–200 ms per molecule
```

| Field | Spec |
|---|---|
| **Module** | `chemvision.models.mol_encoder.MolecularEncoder` |
| **Library** | RDKit 2025.09+ (required), unimol_tools (optional) |
| **Thread safety** | Yes (RDKit is thread-safe for read operations) |

#### 3a. Morgan Fingerprint Encoding

```
encode(smiles: str) → np.ndarray  shape=(2048,) dtype=float32
```

| Parameter | Value |
|---|---|
| **Algorithm** | Morgan / ECFP (Extended Connectivity Fingerprint) |
| **Radius** | 2 (= ECFP4; captures substructures within 4-bond diameter) |
| **Bits** | 2048 |
| **Output** | Binary bit vector cast to float32 |
| **Invalid SMILES** | Returns all-zeros vector (no exception) |

**Tech details:**
```python
RDKit pipeline:
  Chem.MolFromSmiles(smiles)
    → AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    → ConvertToNumpyArray() → np.float32
```

**Uni-Mol2 shim (when `unimol_tools` installed):**
```
_encode_unimol(smiles) → np.ndarray  shape=(512,) dtype=float32

Uses: UniMolRepr(data_type="molecule").get_repr([smiles])
Returns: cls_repr (SE(3)-equivariant 512-dim embedding trained on 800M conformations)
Fallback: Morgan fingerprint on any exception
```

#### 3b. 3D Conformer Generation

```
generate_conformer(smiles: str, seed=42) → ConformerResult
```

| Step | Algorithm | Detail |
|---|---|---|
| 1. Parse | `Chem.MolFromSmiles` + `Chem.AddHs` | Add explicit hydrogens |
| 2. Embed | ETKDGv3 | Distance geometry with experimental torsion-angle preferences |
| 3. Multi-conf | `EmbedMultipleConfs(n=10)` | Generate 10 conformers |
| 4. Minimise | `UFFOptimizeMoleculeConfs(maxIters=1000)` | Universal Force Field energy minimisation |
| 5. Select | `min(energy)` | Keep lowest-energy conformer |

**Output:**
```python
@dataclass
class ConformerResult:
    smiles: str
    atomic_numbers: list[int]              # e.g., [6, 6, 8, 8, 6, 6, ...]
    coordinates: list[list[float]]         # (N_atoms, 3) in Ångströms
    energy_kcal: float | None              # UFF energy in kcal/mol
    success: bool
```

#### 3c. Physicochemical Descriptors

```
compute_descriptors(smiles: str) → MolDescriptors
```

| Descriptor | Source | Unit/Range |
|---|---|---|
| `mw` | `Descriptors.MolWt` | g/mol |
| `logp` | `Descriptors.MolLogP` (Wildman-Crippen) | dimensionless |
| `tpsa` | `Descriptors.TPSA` | Å² |
| `hbd` | `CalcNumHBD` | count |
| `hba` | `CalcNumHBA` | count |
| `rotatable_bonds` | `CalcNumRotatableBonds` | count |
| `rings` | `CalcNumRings` | count |
| `aromatic_rings` | `CalcNumAromaticRings` | count |
| `heavy_atoms` | `GetNumHeavyAtoms` | count |
| `qed` | `QED.qed(mol)` (Bickerton et al. 2012) | [0, 1]; 1 = ideal drug-like |
| `sa_score` | Ertl & Schuffenhauer (2009) via rdkit.Contrib | [1, 10]; 1 = easiest to synthesise |

**Derived property:**
```python
@property
def lipinski_pass(self) -> bool:
    return mw < 500 and logp < 5 and hbd <= 5 and hba <= 10
```

#### 3d. Tanimoto Similarity

```
tanimoto(smiles_a: str, smiles_b: str) → float  [0.0, 1.0]
```

Bit-vector Tanimoto coefficient: |A ∩ B| / |A ∪ B| computed via `DataStructs.TanimotoSimilarity`.

---

### Stage 4: Physics Constraints

```
PURPOSE:  Validate and enrich extracted structural data with hard physical laws
LATENCY:  < 10 ms per call
```

#### 4a. Crystal Symmetry Analysis

| Field | Spec |
|---|---|
| **Module** | `chemvision.physics.symmetry.CrystalSymmetryAnalyzer` |
| **Library** | spglib (C library, Python bindings) |
| **Precision** | `symprec=1e-5` (default; tighten to 1e-3 for noisy structures) |

**Input:**
```python
lattice: np.ndarray              # (3, 3) — rows are lattice vectors in Å
positions: np.ndarray            # (N, 3) — fractional coordinates [0, 1)
atomic_numbers: np.ndarray       # (N,) — Z values (e.g., 29 for Cu)
```

**Output:**
```python
@dataclass
class SymmetryResult:
    space_group_number: int | None       # 1–230
    space_group_symbol: str              # e.g., "Fm-3m"
    crystal_system: str                  # cubic | hexagonal | trigonal | tetragonal |
                                         # orthorhombic | monoclinic | triclinic
    point_group: str                     # e.g., "m-3m"
    hall_symbol: str                     # e.g., "-F 4 2 3"
    wyckoff_letters: list[str]           # e.g., ["a", "a", "a", "a"]
    equivalent_atoms: list[int]          # symmetry-equivalent atom indices
    is_valid: bool                       # False if spglib failed
```

**Convenience method:**
```python
from_lattice_params(
    a, b, c,                             # Å
    alpha=90, beta=90, gamma=90,         # degrees
    species=[29, 29, 29, 29],            # atomic numbers
    fractional_positions=[[0,0,0], ...]  # fractional coords
) → SymmetryResult
```

**Lattice parameter → matrix conversion:**
```
┌ a   0     0   ┐
│ b·cosγ  b·sinγ  0   │
└ c·cosβ  c·cy   c·cz ┘

where cy = (cosα - cosβ·cosγ) / sinγ
      cz = √(1 - cosβ² - cy²)
```

#### 4b. Scherrer Grain-Size Analysis

| Field | Spec |
|---|---|
| **Module** | `chemvision.physics.scherrer.ScherrerAnalyzer` |
| **Equation** | D = K·λ / (β·cos θ) |
| **K** | 0.9 (spherical crystallites, default) |
| **λ** | 1.5406 Å (Cu Kα, default) |

**Input:**
```python
peaks: list[tuple[float, float]]         # [(2θ in degrees, FWHM in degrees), ...]
```

**Output per peak:**
```python
@dataclass
class GrainSizeResult:
    two_theta_deg: float
    fwhm_deg: float
    wavelength_angstrom: float = 1.5406
    scherrer_k: float = 0.9
    grain_size_angstrom: float | None     # D in Å
    grain_size_nm: float | None           # D / 10
    valid: bool                           # False if FWHM ≤ 0
```

**Integration with ExtractSpectrumSkill:**
```python
# After extracting peaks from XRD image:
spectrum: SpectrumData = extract_spectrum_skill(image, model, spectrum_type="XRD")
peaks_for_scherrer = [
    (peak.position, peak.fwhm)
    for peak in spectrum.peaks
    if peak.fwhm is not None and peak.fwhm > 0
]
grain_sizes = ScherrerAnalyzer().analyze_peaks(peaks_for_scherrer)
mean_size = ScherrerAnalyzer().mean_grain_size_nm(peaks_for_scherrer)
```

---

### Stage 5: Property Prediction

```
PURPOSE:  Predict molecular/materials properties from structural data
LATENCY:  5–50 ms (RDKit) | 200–500 ms (MACE-MP-0 on GPU)
```

| Field | Spec |
|---|---|
| **Module** | `chemvision.generation.property_predictor.PropertyPredictor` |
| **Tier 1** | RDKit (always available, CPU) |
| **Tier 2** | MACE-MP-0 universal force field (optional, GPU) |

**Input:**
```python
predict(smiles: str) → PropertyResult                         # Tier 1
predict_crystal(atoms: ase.Atoms) → PropertyResult            # Tier 2
rank_candidates(smiles_list: list[str]) → list[PropertyResult] # Batch + sort
```

**Output:**
```python
@dataclass
class PropertyResult:
    smiles: str

    # ── RDKit tier (always present) ──────────────────────────
    mw: float | None                       # Molecular weight (g/mol)
    logp: float | None                     # Wildman-Crippen LogP
    tpsa: float | None                     # Topological polar surface area (Å²)
    qed: float | None                      # Drug-likeness [0, 1]
    sa_score: float | None                 # Synthetic accessibility [1, 10]
    hbd: int | None                        # H-bond donors
    hba: int | None                        # H-bond acceptors
    rotatable_bonds: int | None

    # ── MACE-MP-0 tier (optional, for crystals) ─────────────
    energy_ev: float | None                # Total potential energy (eV)
    forces_ev_ang: list[list[float]] | None  # Per-atom forces (eV/Å)

    # ── Derived scores ───────────────────────────────────────
    drug_score: float | None               # QED × (1 - SA/10)
    synthesisability: str                  # easy | moderate | hard | very_hard
    backend: str                           # "rdkit" | "mace-mp-0" | "none"
    warnings: list[str]
```

**SA score computation cascade:**
```
1. Try: rdkit.Contrib.SA_Score.sascorer.calculateScore(mol)
2. Fallback: heuristic = 1.0 + rings × 0.5 + stereocenters × 0.8, clamped to [1, 10]
```

**Synthesisability mapping:**
```
sa_score ≤ 3  → "easy"
sa_score ≤ 5  → "moderate"
sa_score ≤ 7  → "hard"
sa_score > 7  → "very_hard"
```

**MACE-MP-0 backend (when `mace-torch` installed):**
```python
from mace.calculators import mace_mp
calc = mace_mp(model="small", dispersion=False, default_dtype="float32", device="cpu")
atoms.calc = calc
energy = atoms.get_potential_energy()    # eV
forces = atoms.get_forces()              # (N, 3) eV/Å
```

| MACE-MP-0 Spec | Value |
|---|---|
| Elements | 89 (H to Ac) |
| Architecture | MACE (higher-order equivariant message passing) |
| Training data | Materials Project trajectories |
| Energy MAE | ~20 meV/atom |
| Forces MAE | ~45 meV/Å |
| No refitting | Works out-of-the-box for any periodic structure |

---

### Stage 6: Generative Optimisation (Pareto MCTS)

```
PURPOSE:  Propose Pareto-optimal molecular analogues via multi-objective tree search
LATENCY:  5–60 s depending on n_iterations and number of objectives
```

| Field | Spec |
|---|---|
| **Module** | `chemvision.generation.pareto_mcts.ParetoMCTS` |
| **Algorithm** | Monte Carlo Tree Search + Pareto dominance ranking |
| **Node selection** | UCB1 (c=1.41) weighted by mean normalised objective values |
| **Expansion** | Atom mutation + fragment addition (validated by RDKit) |
| **Simulation** | 1-step random mutation rollout |
| **Backpropagation** | Cumulative objective scores up to root |

#### Objective definition

```python
@dataclass
class Objective:
    name: str                              # e.g., "qed", "mw_inv", "logp"
    fn: Callable[[str], float]             # SMILES → scalar
    direction: str = "max"                 # "max" or "min"

    def evaluate(smiles) → float:          # normalised: higher = always better
        raw = fn(smiles)
        return raw if direction == "max" else -raw
```

#### Mutation operators

| Operator | Description | Validation |
|---|---|---|
| **Atom substitution** | Replace atom at index i with {C, N, O, F, S} | `Chem.SanitizeMol()` must pass |
| **Fragment addition** | Append from {C(=O)N, C(=O)O, CN, CF, CCl, CS, CO, pyridine, benzene, pyrrolidine, tetrahydrofuran} | Combined SMILES must parse |
| **Heavy-atom cap** | Reject if `GetNumHeavyAtoms() > max_atoms` (default 50) | — |

#### Pareto dominance

```python
def dominates(self, other) -> bool:
    """A dominates B iff A ≥ B in all objectives and A > B in at least one."""
    better_any = False
    for k in self.scores:
        if self.scores[k] < other.scores[k]: return False
        if self.scores[k] > other.scores[k]: better_any = True
    return better_any
```

**Pareto rank:** Number of candidates that dominate this one. Rank 0 = non-dominated front.

#### Full search flow

```
search(seed_smiles, n_iterations=100)
  │
  ├── Create root node for seed_smiles
  ├── Evaluate root on all objectives
  │
  └── FOR i in range(n_iterations):
        │
        ├── SELECT: traverse tree via UCB1 to leaf
        │     ucb1 = (Σ q_values) / (n_obj × visits) + c × √(ln(parent_visits) / visits)
        │
        ├── EXPAND: generate mutations of leaf.smiles
        │     → filter by RDKit validity + heavy-atom cap
        │     → evaluate each child on all objectives
        │
        ├── SIMULATE: random 1-step mutation rollout
        │     → evaluate rollout molecule
        │
        └── BACKPROPAGATE: propagate scores up to root
              cur.visits += 1
              cur.q_values[obj] += scores[obj]

  ├── Collect all evaluated SMILES + scores
  ├── Compute Pareto ranks
  ├── Filter rank-0 candidates (non-dominated)
  ├── Sort by first objective (descending)
  └── Return list[Candidate]
```

**Output:**
```python
@dataclass
class Candidate:
    smiles: str
    scores: dict[str, float]               # {obj_name: normalised_score}
    pareto_rank: int                        # 0 = Pareto front
```

---

### Stage 7: Agent Orchestration (ReAct Loop)

```
PURPOSE:  Chain skills into multi-step scientific reasoning
LATENCY:  5–120 s (1–10 steps × 1–8 s per skill call)
```

| Field | Spec |
|---|---|
| **Module** | `chemvision.agent.agent.ChemVisionAgent` |
| **Pattern** | ReAct (Reason → Act → Observe) with Claude as planner |
| **Max steps** | Configurable (default: 10) |
| **Tool protocol** | Anthropic tool-use API (`tool_use` + `tool_result` blocks) |
| **Extended thinking** | Optional (Sonnet/Opus only, 1K–32K token budget) |

**Planning model configuration:**

| Parameter | Default | Range |
|---|---|---|
| `planning_model` | `claude-sonnet-4-20250514` | Any Claude model ID |
| `max_steps` | 10 | 1–20 |
| `confidence_threshold` | 0.75 | 0.0–1.0 |
| `use_extended_thinking` | False | — |
| `thinking_budget_tokens` | 8000 | 1000–32000 |

**ReAct loop:**
```
messages = [initial_message(question, images)]

for step in range(max_steps):
    │
    ├── response = claude.messages.create(
    │       model=planning_model,
    │       messages=messages,
    │       tools=SKILL_TOOLS + [final_answer_tool],
    │   )
    │
    ├── thought = extract_text(response)         # reasoning text
    ├── thinking = extract_thinking(response)     # extended thinking (if enabled)
    │
    ├── tool_calls = extract_tool_calls(response)
    │
    └── for call in tool_calls:
          │
          ├── if call.name == "final_answer":
          │     answer = call.input["answer"]
          │     BREAK
          │
          ├── skill = registry.get(PLANNER_TO_REGISTRY[call.name])
          ├── result = skill(image, vision_model, **call.input)
          ├── observation = format_observation(result)
          ├── tool_log = ToolCallLog(skill_name, inputs, output, confidence)
          │
          └── messages.append(tool_result_message(call.id, observation))

return AnalysisReport.build(question, paths, answer, tool_logs, steps, threshold)
```

**Planner ↔ Skill name mapping:**
```python
_PLANNER_TO_REGISTRY = {
    "analyze_structure":  "analyze_structure",
    "extract_spectrum":   "extract_spectrum_data",
    "compare_structures": "compare_structures",
    "validate_caption":   "validate_figure_caption",
    "detect_anomaly":     "detect_anomaly",
    "extract_reaction":   "extract_reaction",
    "analyze_microscopy": "analyze_microscopy",
    "identify_molecule":  "molecular_structure",
}
```

**Output:**
```python
class AnalysisReport(BaseModel):
    question: str
    image_paths: list[str]
    final_answer: str                         # synthesised text answer
    tool_logs: list[ToolCallLog]              # per-skill invocation records
    low_confidence_flag: bool                 # True if ANY confidence < threshold
    min_intermediate_confidence: float | None
    num_steps: int
    created_at: datetime
    trace_steps: list[dict]                   # full Thought/Action/Observation trace
    structured_data: dict                     # parsed JSON from final_answer (if valid)
```

---

### Stage 8: Audit & Evaluation

```
PURPOSE:  Benchmark accuracy across task types / difficulties and test robustness
LATENCY:  Minutes to hours (dataset-scale evaluation)
```

#### 8a. Capability Matrix

```
Module: chemvision.audit.matrix.CapabilityMatrix
```

**Dimensions:**
```
                    easy    medium    hard
spatial_reasoning    [ ]      [ ]      [ ]
counting             [ ]      [ ]      [ ]
cross_image_comp.    [ ]      [ ]      [ ]
anomaly_detection    [ ]      [ ]      [ ]
caption_validation   [ ]      [ ]      [ ]
```

**Scoring strategies:**
| Strategy | Logic |
|---|---|
| `exact` | `prediction.strip().lower() == ground_truth.strip().lower()` |
| `substring` | `ground_truth.strip().lower() in prediction.strip().lower()` |

**Output:** Heatmap PNG (matplotlib) + JSON matrix.

#### 8b. Degradation Tester

| Degradation Type | Parameter | Range | Unit |
|---|---|---|---|
| JPEG compression | quality | 10–95 | quality factor |
| Gaussian blur | σ | 0.5–5.0 | pixels |
| Gaussian noise | σ | 0–50 | pixel intensity std |
| Motion blur | kernel_size | 3–31 | pixels |
| Zoom | factor | 0.5–2.0 | ×magnification |

**Binary search algorithm:**
```
lo, hi = clean_param, max_degradation_param
for i in range(n_binary_search_iters):
    mid = (lo + hi) / 2
    acc = evaluate(model, dataset, degrade(mid))
    if acc >= threshold:
        lo = mid       # can tolerate more degradation
    else:
        hi = mid       # accuracy dropped below threshold
return critical_param = mid
```

**Output:**
```python
@dataclass
class ReliabilityEnvelope:
    model_name: str
    threshold: float                         # 0.7 default
    results: dict[str, DegradationResult]    # per degradation type
    evaluated_at: str                        # ISO timestamp
```

---

## 3. Data Schema Reference

### Core Schemas

```python
# ── Input ────────────────────────────────────────────
class ImageRecord(BaseModel):              # chemvision/data/schema.py
    id: str
    image_path: Path
    domain: ImageDomain                    # spectroscopy | microscopy | chromatography |
                                           # molecular_diagram | crystal_structure | simulation | other
    question: str
    answer: str
    bbox: list[float] | None               # normalised [x0, y0, x1, y1]
    difficulty: Literal["easy", "medium", "hard"] | None
    source: str | None
    metadata: dict[str, object]

# ── Intermediate (Agent) ─────────────────────────────
class AgentStep(BaseModel):                # chemvision/agent/trace.py
    step_index: int
    step_type: StepType                    # thought | action | observation | final_answer
    content: str
    skill_name: str | None
    timestamp: datetime

class ToolCallLog(BaseModel):              # chemvision/agent/tool_log.py
    skill_name: str
    inputs: dict[str, Any]
    output_summary: str
    confidence: float | None
    low_confidence: bool
    timestamp: datetime
    raw_output: str

# ── Output (Report) ──────────────────────────────────
class AnalysisReport(BaseModel):           # chemvision/agent/report.py
    question: str
    image_paths: list[str]
    final_answer: str
    tool_logs: list[ToolCallLog]
    low_confidence_flag: bool
    min_intermediate_confidence: float | None
    num_steps: int
    created_at: datetime
    trace_steps: list[dict]
    structured_data: dict
```

### Skill Output Model Hierarchy

```
SkillResult (base)
│   skill_name: str
│   raw_output: str
│   parsed: dict
│   confidence: float | None
│
├── StructureAnalysis
│   ├── LatticeParams {a, b, c, α, β, γ, unit}
│   ├── DefectLocation[] {x, y, defect_type, confidence}
│   ├── symmetry: str
│   └── defect_density: float
│
├── SpectrumData
│   ├── Peak[] {position, intensity, assignment, fwhm}
│   ├── snr: float
│   ├── background_level: float
│   └── spectrum_type: str
│
├── StructureComparison
│   ├── DiffRegion[] {x, y, width, height, description}
│   ├── QuantitativeChange[] {metric, before, after, delta, unit}
│   └── trend: str
│
├── CaptionValidation
│   ├── consistency_score: float
│   └── contradictions: list[str]
│
├── AnomalyReport
│   ├── Anomaly[] {location_x, location_y, anomaly_type, description, severity, confidence}
│   ├── severity: Literal["none"|"low"|"medium"|"high"]
│   └── recommendations: list[str]
│
├── ReactionData
│   ├── Molecule[] {name, smiles, role}
│   ├── ReactionConditions {temperature, pressure, solvent, time, atmosphere, yield_percent}
│   ├── reaction_type: str
│   └── arrow_type: str
│
├── MolecularStructureData
│   ├── smiles, iupac_name, common_name, formula, mw
│   ├── FunctionalGroup[] {name, smarts, count}
│   ├── StereocenterInfo[] {atom_or_bond, descriptor, confidence}
│   ├── ring_systems: list[str]
│   └── num_rings: int
│
├── MicroscopyAnalysis
│   ├── MorphologyInfo {shape, surface_texture, aggregation, description}
│   ├── ParticleMeasurement[] {diameter, aspect_ratio, shape, location_x, location_y}
│   ├── SizeStatistics {mean, std, min, max, unit, distribution, particle_count}
│   ├── ScaleBar {value, unit, pixel_length, nm_per_pixel}
│   ├── imaging_modality: str
│   └── magnification: str
│
└── PropertyPredictionResult
    ├── input_smiles: str
    ├── pubchem_name, pubchem_formula, pubchem_mw, pubchem_logp
    ├── predicted: PropertyResult {mw, logp, tpsa, qed, sa_score, drug_score, ...}
    ├── similar_molecules: list[dict]
    └── OptimisedCandidate[] {smiles, qed, mw, logp, sa_score, drug_score,
                              synthesisability, pareto_rank, scores}
```

---

## 4. Model Backend Reference

### Vision Models

| Model | Type | Params | Input Size | Quantisation | Use Case |
|---|---|---|---|---|---|
| Claude Sonnet 4 | Cloud API | ~70B+ | 5 MB / 8192 px | N/A | Default: best quality, highest latency |
| Claude Opus 4 | Cloud API | ~200B+ | 5 MB / 8192 px | N/A | Complex multi-step reasoning |
| Claude Haiku 4.5 | Cloud API | ~34B | 5 MB / 8192 px | N/A | Fast, cheaper, lower quality |
| LLaVA-1.6-7B | Local GPU | 7.6B | 672×672 | bfloat16 | Offline / air-gapped environments |
| InternVL2-8B | Local GPU | 8B | 448×448 | bfloat16 | Better Chinese-language support |

### Molecular Models

| Model | Type | Input | Output | Availability |
|---|---|---|---|---|
| RDKit Morgan ECFP4 | CPU | SMILES | 2048-dim bit vector | Always (required dep) |
| RDKit ETKDG + UFF | CPU | SMILES | 3D coordinates + energy | Always |
| RDKit Descriptors | CPU | SMILES | ~12 physicochemical values | Always |
| Uni-Mol2 (1.1B) | GPU | SMILES | 512-dim SE(3) embedding | Optional (`unimol_tools`) |
| MACE-MP-0 | GPU/CPU | ASE Atoms | Energy (eV) + Forces (eV/Å) | Optional (`mace-torch`) |

### Fine-Tuning Infrastructure

| Component | Framework | Config |
|---|---|---|
| LoRA adapters | PEFT (Hugging Face) | r=16, α=32, dropout=0.05, targets=[q,v,k,o]_proj |
| Trainer | Hugging Face Trainer | 3 epochs, batch=2, grad_accum=8, lr=2e-4 |
| Logging | W&B | Project: chemvision-lora |
| Config file | `configs/lora_train.yaml` | Full YAML spec |

---

## 5. Deployment Architecture

### Local Development

```bash
# Install
pip install -e ".[dev]"

# Run tests (no network, no GPU, no API key)
pytest tests/test_e2e_new_pipeline.py -m "not network" -v

# Run full test suite
pytest

# Start demo UI
ANTHROPIC_API_KEY=sk-... streamlit run demo/app.py

# Start REST API
ANTHROPIC_API_KEY=sk-... chemvision serve --port 8000 --reload
```

### REST API Endpoints

| Method | Path | Request Body | Response |
|---|---|---|---|
| POST | `/analyze` | `{"question": str, "image_paths": [str]}` | `{"final_answer": str, "tool_calls": [...], "low_confidence_flag": bool, ...}` |
| GET | `/audit` | — | `{"available": bool, "report": dict \| null}` |
| GET | `/health` | — | `{"status": "ok", "version": "0.2.0"}` |

### Process Architecture

```
                    ┌──────────────┐
                    │   Streamlit  │  ← Interactive UI (port 8501)
                    │   demo/app   │
                    └──────┬───────┘
                           │ calls agent.run_stream()
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  HTTP Client │───▶│   FastAPI    │───▶│ ChemVision   │
│  (curl, SDK) │    │  chemvision/ │    │    Agent      │
└──────────────┘    │   api.py     │    └──────┬───────┘
                    └──────────────┘           │
                           ▲                   │ Claude API (HTTPS)
                           │                   ├─────────────────────▶ Anthropic
                     uvicorn (ASGI)             │
                                               │ PubChem (HTTPS)
                                               ├─────────────────────▶ NIH/NLM
                                               │
                                               │ Local compute (CPU/GPU)
                                               ├─ RDKit, spglib, numpy
                                               └─ MACE-MP-0, Uni-Mol2 (optional)
```

---

## 6. Dependency Map

### Required (installed via `pip install -e .`)

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | ≥ 0.40.0 | Claude API client |
| `transformers` | ≥ 4.40.0 | LLaVA / InternVL2 model loading |
| `torch` | ≥ 2.2.0 | PyTorch backbone |
| `peft` | ≥ 0.10.0 | LoRA fine-tuning |
| `rdkit` | ≥ 2024.3.0 | Molecular cheminformatics |
| `spglib` | (system) | Crystal symmetry via C library |
| `chromadb` | ≥ 0.5.0 | Persistent vector store |
| `pillow` | ≥ 10.0.0 | Image I/O |
| `pydantic` | ≥ 2.7.0 | Data validation / schemas |
| `fastapi` | ≥ 0.111.0 | REST API |
| `uvicorn` | ≥ 0.29.0 | ASGI server |
| `streamlit` | ≥ 1.55.0 | Web demo UI |
| `typer` | ≥ 0.12.0 | CLI framework |
| `rich` | ≥ 13.7.0 | Terminal formatting |
| `numpy` | ≥ 1.26.0 | Numerical computing |
| `matplotlib` | ≥ 3.8.0 | Plotting (audit heatmaps, synthetic XRD) |
| `ase` | ≥ 3.23.0 | Atomic Simulation Environment |
| `requests` | ≥ 2.32.0 | PubChem HTTP client |
| `httpx` | ≥ 0.27.0 | Async HTTP (FastAPI test client) |
| `accelerate` | ≥ 0.29.0 | Distributed training |
| `wandb` | ≥ 0.17.0 | Experiment tracking |
| `datasets` | ≥ 2.19.0 | HuggingFace datasets |
| `pandas` | ≥ 2.3.3 | Data manipulation |
| `pypdf` | ≥ 4.2.0 | PDF extraction |
| `pyyaml` | ≥ 6.0 | Config loading |

### Optional (GPU tier)

| Package | Purpose |
|---|---|
| `mace-torch` ≥ 0.3.0 | MACE-MP-0 universal force field |
| `unimol_tools` | Uni-Mol2 SE(3)-equivariant embeddings |

### Dev

| Package | Purpose |
|---|---|
| `pytest` ≥ 8.2.0 | Testing |
| `pytest-asyncio` ≥ 0.23.0 | Async test support |
| `ruff` ≥ 0.4.0 | Linting |
| `mypy` ≥ 1.10.0 | Type checking |

---

## 7. Performance Characteristics

### Latency Budget (typical single-query)

| Stage | Operation | Latency |
|---|---|---|
| 0 | Image load + base64 encode | 10–50 ms |
| 1 | VLM skill call (Claude cloud) | 1–8 s |
| 1 | VLM skill call (LLaVA-7B local) | 3–15 s |
| 2a | PubChem REST fetch | 200–800 ms |
| 2b | Vector store search (1K molecules) | < 1 ms |
| 3a | Morgan fingerprint encoding | 5–10 ms |
| 3b | 3D conformer generation (10 confs + UFF) | 50–200 ms |
| 3c | Descriptor computation | 5–10 ms |
| 4a | spglib symmetry analysis | 1–5 ms |
| 4b | Scherrer grain size (per peak) | < 1 ms |
| 5 | RDKit property prediction | 5–20 ms |
| 5 | MACE-MP-0 (crystal, GPU) | 200–500 ms |
| 6 | Pareto MCTS (80 iterations) | 5–30 s |
| 7 | Full ReAct loop (3–5 steps) | 5–40 s |

**Total typical latency:**
- Simple query (1 skill): **3–10 s**
- Complex query (3–5 skills): **10–40 s**
- With Pareto MCTS optimisation: **15–60 s**

### Memory Footprint

| Component | RAM |
|---|---|
| ChemVisionAgent (no local model) | ~200 MB |
| + LLaVA-7B (bfloat16) | +15 GB |
| + MACE-MP-0 (small) | +500 MB |
| + Uni-Mol2 (1.1B) | +4 GB |
| + ChromaDB (10K molecules) | +50 MB |
| + Vector store numpy (10K molecules) | +80 MB |

### Test Suite

| Suite | Tests | Time | Network | GPU |
|---|---|---|---|---|
| `test_e2e_new_pipeline.py` (offline) | 41 | 21 s | No | No |
| `test_e2e_new_pipeline.py` (full) | 45 | 25 s | Yes | No |
| Full repo (`pytest`) | 265+ | ~2 s* | No | No |

*Original tests are mocked and fast; new pipeline tests are heavier due to RDKit conformer generation and MCTS.
