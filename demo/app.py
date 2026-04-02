"""ChemVision Agent — Streamlit Demo (v2)

A comprehensive scientific analysis platform with five workspaces:

  1. Agent Analysis   — ReAct multi-step reasoning over images
  2. Molecular Lab     — Encode, predict, and optimise molecules from SMILES
  3. Crystal Lab       — Symmetry analysis + Scherrer grain-size from XRD data
  4. Retrieval Explorer — PubChem search + similarity + vector store
  5. Benchmarks        — Audit capability matrix & degradation robustness

Run
---
    ANTHROPIC_API_KEY=sk-... streamlit run demo/app.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── page config (must be first st call) ──────────────────────────────────────

st.set_page_config(
    page_title="ChemVision Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────

_DEMO_DIR = Path(_PROJECT_ROOT) / "demo_images"

_DEMO_PRESETS: list[dict[str, Any]] = [
    {
        "label": "XRD Phase ID",
        "images": ["xrd_200C.png"],
        "question": (
            "What crystal phase is dominant in this XRD pattern? "
            "Extract all peak positions and assign them to crystal planes."
        ),
    },
    {
        "label": "Phase Evolution (3 temps)",
        "images": ["xrd_200C.png", "xrd_400C.png", "xrd_600C.png"],
        "question": (
            "Compare the XRD patterns at 200, 400, and 600 deg C. "
            "Identify any phase transitions and quantify how the dominant peak shifts."
        ),
    },
    {
        "label": "Anomaly Detection",
        "images": ["xrd_200C.png"],
        "question": (
            "Detect any anomalies or unexpected features in this diffraction pattern."
        ),
    },
]

_EXAMPLE_SMILES: dict[str, str] = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Penicillin V": "CC1(C(N2C(S1)C(C2=O)NC(=O)COc3ccccc3)C(=O)O)C",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Artemisinin": "CC1CCC2C(C(=O)OC3CC4(OO3)C(CC2C1C)OC4=O)C",
}

# ── helpers ───────────────────────────────────────────────────────────────────


def _step_icon(step_type: str) -> str:
    return {"thought": "💭", "action": "⚡", "observation": "👁️", "final_answer": "✅"}.get(step_type, "•")


def _confidence_color(conf: float | None) -> str:
    if conf is None:
        return "gray"
    return "green" if conf >= 0.85 else ("orange" if conf >= 0.70 else "red")


def _render_step(step: dict[str, Any]) -> None:
    icon = _step_icon(step["step_type"])
    label = step["step_type"].replace("_", " ").title()
    skill = f" · `{step['skill_name']}`" if step.get("skill_name") else ""
    is_thinking = step.get("content", "").startswith("[Extended Thinking]")
    with st.expander(
        f"{icon} Step {step['step_index']} — {label}{skill}{' 🧠' if is_thinking else ''}",
        expanded=(step["step_type"] == "thought" and not is_thinking),
    ):
        content = step["content"]
        if is_thinking:
            st.caption("Extended thinking — Claude's internal reasoning")
            st.markdown(f"```\n{content.replace('[Extended Thinking]\\n', '', 1)[:2000]}\n```")
        else:
            try:
                st.json(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                st.markdown(f"```\n{content}\n```")


def _render_tool_log(log: dict[str, Any], threshold: float) -> None:
    conf = log.get("confidence")
    color = _confidence_color(conf)
    conf_str = f"{conf:.2f}" if conf is not None else "—"
    low = log.get("low_confidence", False)
    with st.expander(f"🔧 `{log['skill_name']}` — confidence: :{color}[**{conf_str}**]{'  ⚠️' if low else ''}", expanded=low):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Inputs**")
            st.json(log.get("inputs", {}))
        with c2:
            st.markdown("**Output summary**")
            st.text(log.get("output_summary", ""))


def _save_uploads(uploaded_files: list) -> list[str]:
    paths = []
    for f in uploaded_files:
        dest = Path("/tmp") / f.name
        dest.write_bytes(f.read())
        paths.append(str(dest))
    return paths


def _image_to_b64(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


@st.cache_resource(show_spinner=False)
def _get_agent(api_key: str, model: str, threshold: float, max_steps: int, ext_thinking: bool, thinking_budget: int):
    from chemvision.agent.agent import ChemVisionAgent
    from chemvision.agent.config import AgentConfig
    return ChemVisionAgent(AgentConfig(
        anthropic_api_key=api_key, planning_model=model,
        confidence_threshold=threshold, max_steps=max_steps,
        verbose=False, use_extended_thinking=ext_thinking,
        thinking_budget_tokens=thinking_budget,
    ))


@st.cache_resource(show_spinner=False)
def _get_encoder():
    from chemvision.models.mol_encoder import MolecularEncoder
    return MolecularEncoder()


@st.cache_resource(show_spinner=False)
def _get_predictor():
    from chemvision.generation.property_predictor import PropertyPredictor
    return PropertyPredictor(use_mace=False)


@st.cache_resource(show_spinner=False)
def _get_pubchem():
    from chemvision.retrieval.pubchem_client import PubChemClient
    return PubChemClient(timeout=10)


@st.cache_resource(show_spinner=False)
def _get_symmetry():
    from chemvision.physics.symmetry import CrystalSymmetryAnalyzer
    return CrystalSymmetryAnalyzer()


@st.cache_resource(show_spinner=False)
def _get_scherrer():
    from chemvision.physics.scherrer import ScherrerAnalyzer
    return ScherrerAnalyzer()


def _get_vector_store():
    if "vector_store" not in st.session_state:
        from chemvision.retrieval.vector_store import MoleculeVectorStore
        st.session_state["vector_store"] = MoleculeVectorStore()
    return st.session_state["vector_store"]


# ── sidebar: global settings ─────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 ChemVision Agent")
    st.caption("Multimodal Scientific Analysis Platform")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password", placeholder="sk-ant-...",
        help="Required only for Agent Analysis tab. Other tabs run locally.",
    )
    planning_model = st.selectbox(
        "Planning model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
    )
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.75, 0.05)
    max_steps = st.number_input("Max ReAct steps", 1, 20, 10)

    st.divider()
    st.subheader("🧠 Extended Thinking")
    use_extended_thinking = st.toggle("Enable extended thinking", value=False)
    thinking_budget = 8000
    if use_extended_thinking:
        thinking_budget = st.slider("Thinking budget (tokens)", 1000, 32000, 8000, 1000)
        if "haiku" in planning_model:
            st.warning("Extended thinking not supported on Haiku.")

    st.divider()
    st.markdown(
        "**Capabilities**\n"
        "- 9 composable vision skills\n"
        "- RDKit molecular encoder\n"
        "- PubChem retrieval grounding\n"
        "- Pareto MCTS optimisation\n"
        "- spglib crystal symmetry\n"
        "- Scherrer grain-size analysis\n"
    )
    st.caption("ChemVision Agent v0.3 · Powered by Claude + RDKit + spglib")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — 5 TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_agent, tab_mol, tab_crystal, tab_retrieval, tab_bench = st.tabs([
    "🤖 Agent Analysis",
    "🧪 Molecular Lab",
    "💎 Crystal Lab",
    "🔍 Retrieval Explorer",
    "📊 Benchmarks",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AGENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_agent:
    st.header("Agent Analysis")
    st.markdown("Upload scientific images, ask a question, and watch the ReAct agent reason step-by-step.")

    # ── demo presets ──────────────────────────────────────────────────────
    st.subheader("Quick start: load a demo")
    demo_cols = st.columns(len(_DEMO_PRESETS))
    for col, preset in zip(demo_cols, _DEMO_PRESETS):
        with col:
            if st.button(f"📂 {preset['label']}", key=f"demo_{preset['label']}", use_container_width=True):
                paths = [str(_DEMO_DIR / f) for f in preset["images"] if (_DEMO_DIR / f).exists()]
                if paths:
                    st.session_state["demo_image_paths"] = paths
                    st.session_state["demo_image_names"] = preset["images"]
                    st.session_state["question_prefill"] = preset["question"]
                else:
                    st.warning(f"Demo images not found. Run `python gen_demo_images.py` first.")

    # ── image upload ──────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "...or upload images (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True, key="agent_upload",
    )
    if uploaded_files:
        active_image_paths = _save_uploads(uploaded_files)
        active_image_names = [f.name for f in uploaded_files]
        st.session_state.pop("demo_image_paths", None)
    elif "demo_image_paths" in st.session_state:
        active_image_paths = st.session_state["demo_image_paths"]
        active_image_names = st.session_state.get("demo_image_names", [])
    else:
        active_image_paths = []
        active_image_names = []

    if active_image_paths:
        cols = st.columns(min(len(active_image_paths), 4))
        for col, path, name in zip(cols, active_image_paths, active_image_names):
            with col:
                st.image(path, caption=name, use_container_width=True)

    # ── question + run ────────────────────────────────────────────────────
    prefill = st.session_state.pop("question_prefill", "")
    question = st.text_area("Question", value=prefill, height=80, label_visibility="collapsed",
                            placeholder="e.g. What phase transition occurs between these XRD patterns?")
    run_disabled = not (api_key and active_image_paths and question.strip())
    run_clicked = st.button("▶ Run Agent", type="primary", disabled=run_disabled,
                            use_container_width=True, key="run_agent")
    if run_disabled and not api_key:
        st.info("Enter your Anthropic API key in the sidebar to use the agent.")

    # ── streaming execution ───────────────────────────────────────────────
    if run_clicked:
        agent = _get_agent(api_key, planning_model, confidence_threshold, int(max_steps),
                           use_extended_thinking, thinking_budget)
        t0 = time.perf_counter()
        with st.status("Agent is thinking...", expanded=True) as status:
            steps_rendered: list[dict[str, Any]] = []
            report = None
            try:
                from chemvision.agent.report import AnalysisReport
                for event in agent.run_stream(question=question.strip(), image_paths=active_image_paths):
                    if isinstance(event, AnalysisReport):
                        report = event
                    else:
                        step_dict = event.model_dump(mode="json")
                        steps_rendered.append(step_dict)
                        _render_step(step_dict)
            except Exception as exc:
                status.update(label="Agent error", state="error")
                st.error(f"Agent error: {exc}")
                st.stop()
            elapsed = time.perf_counter() - t0
            status.update(label=f"Done in {elapsed:.1f}s — {len(steps_rendered)} steps", state="complete", expanded=False)

        if report is None:
            st.error("Agent completed without returning a report.")
            st.stop()
        st.session_state["last_report"] = report
        st.session_state["last_elapsed"] = elapsed

    # ── results ───────────────────────────────────────────────────────────
    if "last_report" in st.session_state:
        report = st.session_state["last_report"]
        elapsed = st.session_state.get("last_elapsed", 0.0)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Steps", report.num_steps)
        m2.metric("Skill calls", len(report.tool_logs))
        min_conf = report.min_intermediate_confidence
        m3.metric("Min confidence", f"{min_conf:.2f}" if min_conf is not None else "—")
        m4.metric("Latency", f"{elapsed:.1f}s")

        if report.low_confidence_flag:
            st.warning(f"⚠️ Some skill calls below confidence threshold ({confidence_threshold:.2f}).")

        tab_a, tab_t, tab_l = st.tabs(["📊 Answer", "🔍 Reasoning Trace", "🔧 Tool Log"])
        with tab_a:
            if report.structured_data:
                st.json(report.structured_data)
            else:
                st.markdown(report.final_answer)
        with tab_t:
            if report.trace_steps:
                for step in report.trace_steps:
                    _render_step(step)
            else:
                st.info("No trace steps recorded.")
        with tab_l:
            for log in report.tool_logs:
                _render_tool_log(log.model_dump(), confidence_threshold)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MOLECULAR LAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_mol:
    st.header("Molecular Lab")
    st.markdown(
        "Encode molecules, predict properties, and run multi-objective Pareto MCTS optimisation. "
        "**No API key required** — runs entirely on CPU with RDKit."
    )

    # ── input ─────────────────────────────────────────────────────────────
    col_input, col_preset = st.columns([3, 1])
    with col_input:
        mol_smiles = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="mol_smiles",
                                   placeholder="Enter a SMILES string")
    with col_preset:
        preset_name = st.selectbox("Or pick a molecule", ["(custom)"] + list(_EXAMPLE_SMILES.keys()), key="mol_preset")
        if preset_name != "(custom)":
            mol_smiles = _EXAMPLE_SMILES[preset_name]

    if not mol_smiles.strip():
        st.info("Enter a SMILES string above to begin.")
        st.stop()

    # ── validate SMILES ───────────────────────────────────────────────────
    from rdkit import Chem
    mol_obj = Chem.MolFromSmiles(mol_smiles)
    if mol_obj is None:
        st.error(f"Invalid SMILES: `{mol_smiles}`")
    else:
        canonical = Chem.MolToSmiles(mol_obj)
        st.success(f"Valid molecule: **{canonical}** ({mol_obj.GetNumHeavyAtoms()} heavy atoms, {Chem.rdMolDescriptors.CalcNumRings(mol_obj)} rings)")

        # ── sub-tabs ──────────────────────────────────────────────────────
        mol_tab1, mol_tab2, mol_tab3, mol_tab4 = st.tabs([
            "📋 Descriptors & Fingerprint",
            "🧬 3D Conformer",
            "💊 Property Prediction",
            "🎯 Pareto MCTS Optimisation",
        ])

        # ── descriptors & fingerprint ─────────────────────────────────────
        with mol_tab1:
            enc = _get_encoder()
            desc = enc.compute_descriptors(mol_smiles)
            emb = enc.encode(mol_smiles)

            st.subheader("Physicochemical Descriptors")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("MW", f"{desc.mw:.1f} g/mol" if desc.mw else "—")
            d2.metric("LogP", f"{desc.logp:.2f}" if desc.logp is not None else "—")
            d3.metric("TPSA", f"{desc.tpsa:.1f} A²" if desc.tpsa else "—")
            d4.metric("QED", f"{desc.qed:.3f}" if desc.qed else "—")
            d5.metric("Lipinski", "PASS" if desc.lipinski_pass else ("FAIL" if desc.lipinski_pass is False else "—"))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("HBD", desc.hbd if desc.hbd is not None else "—")
            c2.metric("HBA", desc.hba if desc.hba is not None else "—")
            c3.metric("Rotatable bonds", desc.rotatable_bonds if desc.rotatable_bonds is not None else "—")
            c4.metric("Rings", desc.rings if desc.rings is not None else "—")
            c5.metric("Aromatic rings", desc.aromatic_rings if desc.aromatic_rings is not None else "—")

            st.divider()
            st.subheader("Morgan Fingerprint (ECFP4, 2048-bit)")
            st.caption(f"Non-zero bits: {int(emb.sum())} / 2048 ({emb.sum()/20.48:.1f}% density)")

            # Fingerprint heatmap
            fp_img = emb.reshape(32, 64)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.imshow(fp_img, aspect="auto", cmap="Blues", interpolation="nearest")
            ax.set_xlabel("Bit index (mod 64)")
            ax.set_ylabel("Row")
            ax.set_title("Morgan ECFP4 bit vector")
            st.pyplot(fig)
            plt.close()

        # ── 3D conformer ──────────────────────────────────────────────────
        with mol_tab2:
            st.subheader("3D Conformer (ETKDG + UFF)")
            if st.button("Generate conformer", key="gen_conf", type="primary"):
                with st.spinner("Running ETKDG embedding + UFF minimisation..."):
                    enc = _get_encoder()
                    conf = enc.generate_conformer(mol_smiles)
                if conf.success:
                    st.success(f"Conformer generated: {conf.num_atoms} atoms, energy = {conf.energy_kcal:.1f} kcal/mol")

                    coords = np.array(conf.coordinates)
                    df_atoms = pd.DataFrame({
                        "Atom #": list(range(conf.num_atoms)),
                        "Z": conf.atomic_numbers,
                        "x (A)": coords[:, 0].round(3),
                        "y (A)": coords[:, 1].round(3),
                        "z (A)": coords[:, 2].round(3),
                    })
                    st.dataframe(df_atoms, use_container_width=True, height=300)

                    # 3D scatter plot
                    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br"}
                    labels = [elem_map.get(z, str(z)) for z in conf.atomic_numbers]
                    df_3d = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "element": labels})
                    st.scatter_chart(df_3d, x="x", y="y", color="element", size=80)
                    st.caption("2D projection (x vs y) — atom colours by element")
                else:
                    st.error("Conformer generation failed for this molecule.")

        # ── property prediction ───────────────────────────────────────────
        with mol_tab3:
            st.subheader("Property Prediction (RDKit)")
            pred = _get_predictor()
            result = pred.predict(mol_smiles)

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Drug-likeness**")
                p1, p2 = st.columns(2)
                p1.metric("QED", f"{result.qed:.3f}" if result.qed else "—")
                p2.metric("Drug score", f"{result.drug_score:.3f}" if result.drug_score else "—")

                st.markdown("**Synthesisability**")
                s1, s2 = st.columns(2)
                s1.metric("SA score", f"{result.sa_score:.1f}" if result.sa_score else "—",
                          help="1 = easy, 10 = very hard to synthesise")
                s2.metric("Label", result.synthesisability.upper())

            with col_r:
                st.markdown("**ADMET Descriptors**")
                props = {
                    "MW (g/mol)": f"{result.mw:.1f}" if result.mw else "—",
                    "LogP": f"{result.logp:.2f}" if result.logp is not None else "—",
                    "TPSA (A²)": f"{result.tpsa:.1f}" if result.tpsa else "—",
                    "HBD": result.hbd if result.hbd is not None else "—",
                    "HBA": result.hba if result.hba is not None else "—",
                    "Rotatable bonds": result.rotatable_bonds if result.rotatable_bonds is not None else "—",
                }
                st.dataframe(pd.DataFrame(props.items(), columns=["Property", "Value"]),
                             use_container_width=True, hide_index=True)

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

        # ── Pareto MCTS ───────────────────────────────────────────────────
        with mol_tab4:
            st.subheader("Pareto MCTS Multi-Objective Optimisation")
            st.markdown(
                "Explore molecular analogues that optimise multiple objectives simultaneously. "
                "The search uses Monte Carlo Tree Search with Pareto dominance ranking."
            )

            mc1, mc2 = st.columns(2)
            with mc1:
                n_iter = st.slider("MCTS iterations", 10, 200, 50, 10, key="mcts_iter",
                                   help="More iterations = larger search space explored")
            with mc2:
                max_atoms = st.slider("Max heavy atoms", 10, 80, 50, 5, key="mcts_atoms")

            st.markdown("**Objectives** (all active):")
            st.markdown("- **QED** (maximise) — drug-likeness\n"
                        "- **1/MW** (maximise) — prefer smaller molecules\n"
                        "- **LogP closeness to 2** (maximise) — ideal lipophilicity window")

            if st.button("🚀 Run Pareto MCTS", type="primary", key="run_mcts"):
                from chemvision.generation.pareto_mcts import ParetoMCTS, Objective

                pred_fn = _get_predictor()
                objectives = [
                    Objective("QED", fn=lambda s: pred_fn.predict(s).qed or 0.0, direction="max"),
                    Objective("1/MW", fn=lambda s: 1.0 / max(pred_fn.predict(s).mw or 500, 1), direction="max"),
                    Objective("LogP~2", fn=lambda s: -abs((pred_fn.predict(s).logp or 5) - 2.0), direction="max"),
                ]
                mcts = ParetoMCTS(objectives, max_atoms=max_atoms, seed=42)

                with st.spinner(f"Running MCTS ({n_iter} iterations)..."):
                    t0 = time.perf_counter()
                    front = mcts.search(mol_smiles, n_iterations=n_iter)
                    elapsed = time.perf_counter() - t0

                st.success(f"Found **{len(front)}** Pareto-optimal candidates in {elapsed:.1f}s")

                # Build results table
                rows = []
                for c in front[:20]:
                    p = pred_fn.predict(c.smiles)
                    rows.append({
                        "SMILES": c.smiles,
                        "QED": round(p.qed, 3) if p.qed else None,
                        "MW": round(p.mw, 1) if p.mw else None,
                        "LogP": round(p.logp, 2) if p.logp is not None else None,
                        "SA score": round(p.sa_score, 1) if p.sa_score else None,
                        "Drug score": round(p.drug_score, 3) if p.drug_score else None,
                        "Synth.": p.synthesisability,
                        "Pareto rank": c.pareto_rank,
                    })
                df_front = pd.DataFrame(rows)
                st.dataframe(df_front, use_container_width=True, hide_index=True)

                # Pareto front scatter
                if len(rows) > 1 and all(r.get("QED") is not None and r.get("MW") is not None for r in rows):
                    st.markdown("**Pareto Front: QED vs MW**")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    qed_vals = [r["QED"] for r in rows if r["QED"]]
                    mw_vals = [r["MW"] for r in rows if r["MW"]]
                    ax2.scatter(mw_vals[:len(qed_vals)], qed_vals, c="steelblue", s=60, alpha=0.7, edgecolors="navy")
                    # Mark the seed molecule
                    seed_p = pred_fn.predict(mol_smiles)
                    if seed_p.mw and seed_p.qed:
                        ax2.scatter([seed_p.mw], [seed_p.qed], c="red", s=120, marker="*", zorder=5, label="Seed")
                        ax2.legend()
                    ax2.set_xlabel("Molecular Weight (g/mol)")
                    ax2.set_ylabel("QED (drug-likeness)")
                    ax2.set_title("Pareto Front")
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CRYSTAL LAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_crystal:
    st.header("Crystal Lab")
    st.markdown(
        "Analyse crystal symmetry via spglib and compute grain sizes from XRD peak broadening "
        "via the Scherrer equation. **No API key required.**"
    )

    crys_tab1, crys_tab2 = st.tabs(["💎 Symmetry Analysis", "📐 Scherrer Grain Size"])

    # ── symmetry ──────────────────────────────────────────────────────────
    with crys_tab1:
        st.subheader("Crystal Symmetry (spglib)")
        st.markdown("Enter lattice parameters and atomic positions to determine the space group.")

        # Preset structures
        preset_crystal = st.selectbox("Preset structures", [
            "(custom)", "FCC Cu (Fm-3m)", "BCC Fe (Im-3m)", "Rutile TiO2 (P4_2/mnm)",
            "Diamond Si (Fd-3m)", "Rocksalt NaCl (Fm-3m)",
        ], key="crystal_preset")

        _CRYSTAL_PRESETS = {
            "FCC Cu (Fm-3m)": {
                "a": 3.615, "b": 3.615, "c": 3.615, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "positions": "0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5",
                "numbers": "29 29 29 29",
            },
            "BCC Fe (Im-3m)": {
                "a": 2.87, "b": 2.87, "c": 2.87, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "positions": "0.0 0.0 0.0\n0.5 0.5 0.5",
                "numbers": "26 26",
            },
            "Rutile TiO2 (P4_2/mnm)": {
                "a": 4.594, "b": 4.594, "c": 2.959, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "positions": "0.0 0.0 0.0\n0.5 0.5 0.5\n0.3053 0.3053 0.0\n0.6947 0.6947 0.0\n0.8053 0.1947 0.5\n0.1947 0.8053 0.5",
                "numbers": "22 22 8 8 8 8",
            },
            "Diamond Si (Fd-3m)": {
                "a": 5.431, "b": 5.431, "c": 5.431, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "positions": "0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5\n0.25 0.25 0.25\n0.75 0.75 0.25\n0.75 0.25 0.75\n0.25 0.75 0.75",
                "numbers": "14 14 14 14 14 14 14 14",
            },
            "Rocksalt NaCl (Fm-3m)": {
                "a": 5.64, "b": 5.64, "c": 5.64, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "positions": "0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5\n0.5 0.5 0.5\n0.0 0.0 0.5\n0.0 0.5 0.0\n0.5 0.0 0.0",
                "numbers": "11 11 11 11 17 17 17 17",
            },
        }

        # Defaults
        defaults = _CRYSTAL_PRESETS.get(preset_crystal, {
            "a": 3.615, "b": 3.615, "c": 3.615, "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
            "positions": "0.0 0.0 0.0", "numbers": "14",
        })

        st.markdown("**Lattice parameters**")
        lc1, lc2, lc3, la1, la2, la3 = st.columns(6)
        a_val = lc1.number_input("a (A)", value=defaults["a"], format="%.3f", key="sym_a")
        b_val = lc2.number_input("b (A)", value=defaults["b"], format="%.3f", key="sym_b")
        c_val = lc3.number_input("c (A)", value=defaults["c"], format="%.3f", key="sym_c")
        alpha_val = la1.number_input("alpha (deg)", value=defaults["alpha"], format="%.1f", key="sym_alpha")
        beta_val = la2.number_input("beta (deg)", value=defaults["beta"], format="%.1f", key="sym_beta")
        gamma_val = la3.number_input("gamma (deg)", value=defaults["gamma"], format="%.1f", key="sym_gamma")

        st.markdown("**Atomic positions** (fractional, one atom per line: `x y z`)")
        positions_text = st.text_area("Positions", value=defaults["positions"], height=120, key="sym_pos", label_visibility="collapsed")
        st.markdown("**Atomic numbers** (space-separated, matching position count)")
        numbers_text = st.text_input("Atomic numbers", value=defaults["numbers"], key="sym_nums", label_visibility="collapsed")

        if st.button("Analyze symmetry", type="primary", key="run_sym"):
            try:
                positions = [[float(x) for x in line.split()] for line in positions_text.strip().split("\n") if line.strip()]
                numbers = [int(x) for x in numbers_text.strip().split()]

                if len(positions) != len(numbers):
                    st.error(f"Position count ({len(positions)}) != atomic number count ({len(numbers)})")
                else:
                    analyzer = _get_symmetry()
                    sym_result = analyzer.from_lattice_params(
                        a_val, b_val, c_val, alpha_val, beta_val, gamma_val,
                        species=numbers, fractional_positions=positions,
                    )

                    if sym_result.is_valid:
                        st.success(f"**{sym_result.summary}**")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Space group #", sym_result.space_group_number)
                        r2.metric("Symbol", sym_result.space_group_symbol)
                        r3.metric("Crystal system", sym_result.crystal_system.capitalize())
                        r4.metric("Point group", sym_result.point_group)

                        if sym_result.wyckoff_letters:
                            st.markdown(f"**Wyckoff letters:** {', '.join(str(w) for w in sym_result.wyckoff_letters)}")
                        if sym_result.hall_symbol:
                            st.caption(f"Hall symbol: {sym_result.hall_symbol}")
                    else:
                        st.error("Symmetry analysis failed — check that the structure is physically valid.")
            except Exception as exc:
                st.error(f"Parse error: {exc}")

    # ── Scherrer ──────────────────────────────────────────────────────────
    with crys_tab2:
        st.subheader("Scherrer Grain-Size Estimation")
        st.markdown("Compute crystallite size from XRD peak broadening: **D = K*lambda / (beta * cos theta)**")

        sc1, sc2 = st.columns(2)
        with sc1:
            wavelength = st.number_input("Wavelength (A)", value=1.5406, format="%.4f", key="sch_wl",
                                         help="1.5406 A = Cu Ka")
            scherrer_k = st.number_input("Scherrer K", value=0.9, format="%.2f", key="sch_k",
                                         help="0.9 for spherical crystallites")
        with sc2:
            st.markdown("**Preset XRD peak sets**")
            sch_preset = st.selectbox("Presets", [
                "(custom)", "TiO2 anatase", "TiO2 rutile", "Al2O3 corundum",
            ], key="sch_preset")
            _SCH_PRESETS = {
                "TiO2 anatase": "25.3 0.35\n48.0 0.42\n53.8 0.45\n55.1 0.50",
                "TiO2 rutile": "27.4 0.30\n36.1 0.38\n41.2 0.40\n54.3 0.48",
                "Al2O3 corundum": "25.6 0.28\n35.1 0.32\n43.3 0.35\n52.5 0.40\n57.5 0.42",
            }

        peak_default = _SCH_PRESETS.get(sch_preset, "25.3 0.35\n48.0 0.42")
        st.markdown("**Peaks** (one per line: `2theta(deg)  FWHM(deg)`)")
        peaks_text = st.text_area("Peaks", value=peak_default, height=120, key="sch_peaks", label_visibility="collapsed")

        if st.button("Calculate grain sizes", type="primary", key="run_sch"):
            try:
                peaks = []
                for line in peaks_text.strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        peaks.append((float(parts[0]), float(parts[1])))

                scherrer = _get_scherrer()
                # Override wavelength/K if changed
                scherrer.wavelength = wavelength
                scherrer.scherrer_k = scherrer_k

                results = scherrer.analyze_peaks(peaks)
                mean_size = scherrer.mean_grain_size_nm(peaks)

                if mean_size is not None:
                    st.success(f"**Mean crystallite size: {mean_size:.1f} nm**")

                rows = []
                for r in results:
                    rows.append({
                        "2theta (deg)": r.two_theta_deg,
                        "FWHM (deg)": r.fwhm_deg,
                        "Grain size (nm)": round(r.grain_size_nm, 1) if r.grain_size_nm else "—",
                        "Grain size (A)": round(r.grain_size_angstrom, 0) if r.grain_size_angstrom else "—",
                        "Valid": "Yes" if r.valid else "No",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Bar chart
                valid_results = [r for r in results if r.valid and r.grain_size_nm]
                if len(valid_results) > 1:
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    x = [f"{r.two_theta_deg:.1f}" for r in valid_results]
                    y = [r.grain_size_nm for r in valid_results]
                    bars = ax3.bar(x, y, color="steelblue", edgecolor="navy")
                    ax3.axhline(mean_size, color="red", linestyle="--", label=f"Mean = {mean_size:.1f} nm")
                    ax3.set_xlabel("2theta (deg)")
                    ax3.set_ylabel("Grain size (nm)")
                    ax3.set_title("Scherrer grain size per peak")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig3)
                    plt.close()
            except Exception as exc:
                st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RETRIEVAL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

with tab_retrieval:
    st.header("Retrieval Explorer")
    st.markdown(
        "Search PubChem for compound properties, find similar molecules by Tanimoto similarity, "
        "and manage the local vector store. **No API key required** (uses free PubChem REST API)."
    )

    ret_tab1, ret_tab2, ret_tab3 = st.tabs([
        "🔎 PubChem Lookup",
        "🧬 Similarity Search",
        "💾 Vector Store",
    ])

    # ── PubChem lookup ────────────────────────────────────────────────────
    with ret_tab1:
        st.subheader("PubChem Compound Lookup")
        lk_col1, lk_col2 = st.columns([3, 1])
        with lk_col1:
            lookup_query = st.text_input("Search by name or SMILES", value="aspirin", key="pc_lookup")
        with lk_col2:
            lookup_mode = st.radio("Mode", ["name", "SMILES"], horizontal=True, key="pc_mode")

        if st.button("Search PubChem", type="primary", key="run_pc_lookup"):
            client = _get_pubchem()
            with st.spinner("Querying PubChem..."):
                if lookup_mode == "name":
                    props = client.fetch_by_name(lookup_query)
                else:
                    props = client.fetch_by_smiles(lookup_query)

            if props:
                st.success(f"Found: **{props.get('IUPACName', props.get('CanonicalSMILES', '?'))}**")

                # Display as card
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("CID", props.get("CID", "—"))
                p2.metric("Formula", props.get("MolecularFormula", "—"))
                p3.metric("MW", f"{props.get('MolecularWeight', '—')}")
                p4.metric("XLogP", f"{props.get('XLogP', '—')}")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric("TPSA", f"{props.get('TPSA', '—')}")
                q2.metric("HBD", props.get("HBondDonorCount", "—"))
                q3.metric("HBA", props.get("HBondAcceptorCount", "—"))
                q4.metric("Complexity", f"{props.get('Complexity', '—')}")

                can_smi = props.get("CanonicalSMILES")
                if can_smi:
                    st.code(can_smi, language=None)
                    st.caption("Canonical SMILES")

                with st.expander("Full PubChem response"):
                    st.json(props)
            else:
                st.warning("No results found. Check the query or try a different search mode.")

    # ── similarity search ─────────────────────────────────────────────────
    with ret_tab2:
        st.subheader("Tanimoto Similarity Search")
        st.markdown("Find structurally similar compounds in PubChem (2D fingerprint similarity).")

        sim_col1, sim_col2, sim_col3 = st.columns([3, 1, 1])
        with sim_col1:
            sim_smiles = st.text_input("Query SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="sim_smi")
        with sim_col2:
            sim_threshold = st.number_input("Threshold", 50, 99, 90, key="sim_thresh")
        with sim_col3:
            sim_max = st.number_input("Max results", 1, 20, 5, key="sim_max")

        if st.button("Find similar compounds", type="primary", key="run_sim"):
            client = _get_pubchem()
            with st.spinner("Searching PubChem..."):
                results = client.get_similar_compounds(sim_smiles, threshold=sim_threshold, max_results=sim_max)

            if results:
                st.success(f"Found **{len(results)}** similar compounds (Tanimoto >= {sim_threshold}%)")
                rows = []
                for r in results:
                    rows.append({
                        "CID": r.get("CID", "—"),
                        "IUPAC Name": r.get("IUPACName", "—"),
                        "Formula": r.get("MolecularFormula", "—"),
                        "MW": r.get("MolecularWeight", "—"),
                        "XLogP": r.get("XLogP", "—"),
                        "SMILES": r.get("CanonicalSMILES", "—"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No similar compounds found (PubChem may be unreachable or threshold too high).")

        st.divider()
        st.subheader("Pairwise Tanimoto")
        st.markdown("Compare two molecules directly.")
        pw_c1, pw_c2 = st.columns(2)
        with pw_c1:
            pw_a = st.text_input("Molecule A", value="CC(=O)Oc1ccccc1C(=O)O", key="pw_a")
        with pw_c2:
            pw_b = st.text_input("Molecule B", value="CC(C)Cc1ccc(cc1)C(C)C(=O)O", key="pw_b")

        if st.button("Compute Tanimoto", key="run_pw"):
            enc = _get_encoder()
            sim = enc.tanimoto(pw_a, pw_b)
            st.metric("Tanimoto coefficient", f"{sim:.4f}")
            if sim >= 0.85:
                st.success("Very high similarity — likely same scaffold.")
            elif sim >= 0.5:
                st.info("Moderate similarity — related scaffolds.")
            else:
                st.warning("Low similarity — structurally distinct.")

    # ── vector store ──────────────────────────────────────────────────────
    with ret_tab3:
        st.subheader("Local Vector Store")
        st.markdown("Build an in-memory molecule database for fast cosine-similarity retrieval.")

        store = _get_vector_store()
        enc = _get_encoder()

        st.metric("Molecules stored", len(store))

        vs_c1, vs_c2 = st.columns([3, 1])
        with vs_c1:
            vs_add_smi = st.text_input("Add molecule (SMILES)", key="vs_add")
        with vs_c2:
            vs_add_name = st.text_input("Name (optional)", key="vs_name")

        if st.button("Add to store", key="vs_add_btn"):
            if vs_add_smi.strip():
                emb = enc.encode(vs_add_smi.strip())
                name = vs_add_name.strip() or vs_add_smi.strip()
                store.add(name, emb, {"smiles": vs_add_smi.strip()})
                st.success(f"Added **{name}** to vector store ({len(store)} total)")
                st.rerun()

        # Bulk add presets
        if st.button("Add all example molecules", key="vs_bulk"):
            for name, smi in _EXAMPLE_SMILES.items():
                emb = enc.encode(smi)
                store.add(name, emb, {"smiles": smi})
            st.success(f"Added {len(_EXAMPLE_SMILES)} molecules ({len(store)} total)")
            st.rerun()

        if len(store) > 0:
            st.divider()
            st.markdown("**Search the store**")
            vs_query = st.text_input("Query SMILES", value="CC(=O)Oc1ccccc1C(=O)O", key="vs_query")
            vs_k = st.slider("Top-k results", 1, min(20, len(store)), min(5, len(store)), key="vs_k")

            if st.button("Search", key="vs_search"):
                query_emb = enc.encode(vs_query)
                hits = store.search(query_emb, k=vs_k)
                if hits:
                    df_hits = pd.DataFrame(hits)
                    st.dataframe(df_hits, use_container_width=True, hide_index=True)
                else:
                    st.info("No results.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

with tab_bench:
    st.header("Benchmarks & System Info")
    st.markdown("Run local benchmarks and inspect system capabilities.")

    bench_tab1, bench_tab2, bench_tab3 = st.tabs([
        "⚡ Pipeline Benchmark",
        "📦 Registered Skills",
        "🔧 System Info",
    ])

    # ── pipeline benchmark ────────────────────────────────────────────────
    with bench_tab1:
        st.subheader("End-to-End Pipeline Benchmark")
        st.markdown(
            "Time each stage of the scientific pipeline on a set of reference molecules. "
            "No network or GPU required."
        )

        bench_mols = st.multiselect(
            "Molecules to benchmark",
            list(_EXAMPLE_SMILES.keys()),
            default=["Aspirin", "Caffeine", "Ibuprofen", "Paracetamol"],
            key="bench_mols",
        )

        if st.button("Run benchmark", type="primary", key="run_bench"):
            smiles_list = [_EXAMPLE_SMILES[m] for m in bench_mols]
            timings: dict[str, float] = {}

            enc = _get_encoder()
            pred = _get_predictor()

            # 1. Encoding
            t0 = time.perf_counter()
            embeddings = enc.encode_batch(smiles_list)
            timings["Morgan encoding"] = time.perf_counter() - t0

            # 2. Conformer generation
            t0 = time.perf_counter()
            conformers = [enc.generate_conformer(s) for s in smiles_list]
            timings["3D conformer (ETKDG+UFF)"] = time.perf_counter() - t0

            # 3. Descriptor computation
            t0 = time.perf_counter()
            descriptors = [enc.compute_descriptors(s) for s in smiles_list]
            timings["Descriptor computation"] = time.perf_counter() - t0

            # 4. Property prediction
            t0 = time.perf_counter()
            predictions = [pred.predict(s) for s in smiles_list]
            timings["Property prediction (RDKit)"] = time.perf_counter() - t0

            # 5. Vector store insert + search
            t0 = time.perf_counter()
            from chemvision.retrieval.vector_store import MoleculeVectorStore
            bench_store = MoleculeVectorStore()
            for i, (name, emb) in enumerate(zip(bench_mols, embeddings)):
                bench_store.add(name, emb)
            _ = bench_store.search(embeddings[0], k=3)
            timings["Vector store (insert+search)"] = time.perf_counter() - t0

            # 6. Symmetry analysis
            t0 = time.perf_counter()
            sym = _get_symmetry()
            _ = sym.from_lattice_params(3.615, 3.615, 3.615, 90, 90, 90,
                                        species=[29, 29, 29, 29],
                                        fractional_positions=[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
            timings["Symmetry analysis (spglib)"] = time.perf_counter() - t0

            # 7. Scherrer
            t0 = time.perf_counter()
            sch = _get_scherrer()
            _ = sch.analyze_peaks([(25.3, 0.35), (48.0, 0.42), (53.8, 0.45)])
            timings["Scherrer grain size"] = time.perf_counter() - t0

            # 8. Pareto MCTS (small run)
            t0 = time.perf_counter()
            from chemvision.generation.pareto_mcts import ParetoMCTS, Objective
            objectives = [
                Objective("qed", fn=lambda s: pred.predict(s).qed or 0, direction="max"),
                Objective("mw", fn=lambda s: -(pred.predict(s).mw or 500), direction="max"),
            ]
            front = ParetoMCTS(objectives, seed=42).search(smiles_list[0], n_iterations=20)
            timings["Pareto MCTS (20 iter)"] = time.perf_counter() - t0

            total = sum(timings.values())
            st.success(f"**Total pipeline: {total:.2f}s** across {len(bench_mols)} molecules")

            # Results table
            df_timing = pd.DataFrame([
                {"Stage": k, "Time (ms)": round(v * 1000, 1), "% of total": round(v / total * 100, 1)}
                for k, v in timings.items()
            ])
            st.dataframe(df_timing, use_container_width=True, hide_index=True)

            # Bar chart
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            bars = ax4.barh(list(timings.keys()), [v * 1000 for v in timings.values()], color="steelblue", edgecolor="navy")
            ax4.set_xlabel("Time (ms)")
            ax4.set_title(f"Pipeline latency ({len(bench_mols)} molecules)")
            ax4.invert_yaxis()
            ax4.grid(True, alpha=0.3, axis="x")
            for bar, v in zip(bars, timings.values()):
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                         f"{v*1000:.0f} ms", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            # Molecule property table
            st.divider()
            st.subheader("Molecule Properties")
            prop_rows = []
            for name, smi, p, d, c in zip(bench_mols, smiles_list, predictions, descriptors, conformers):
                prop_rows.append({
                    "Name": name,
                    "SMILES": smi,
                    "MW": round(p.mw, 1) if p.mw else "—",
                    "QED": round(p.qed, 3) if p.qed else "—",
                    "LogP": round(p.logp, 2) if p.logp is not None else "—",
                    "SA score": round(p.sa_score, 1) if p.sa_score else "—",
                    "Synth.": p.synthesisability,
                    "Lipinski": "PASS" if d.lipinski_pass else "FAIL",
                    "3D atoms": c.num_atoms if c.success else "—",
                    "UFF energy": f"{c.energy_kcal:.0f}" if c.energy_kcal else "—",
                })
            st.dataframe(pd.DataFrame(prop_rows), use_container_width=True, hide_index=True)

    # ── registered skills ─────────────────────────────────────────────────
    with bench_tab2:
        st.subheader("Registered Skills")
        from chemvision.skills.skill_registry import DEFAULT_REGISTRY

        skill_names = DEFAULT_REGISTRY.list_skills()
        st.metric("Total skills", len(skill_names))

        skill_info = {
            "analyze_structure": ("Crystallographic lattice params, symmetry, defects", "StructureAnalysis"),
            "extract_spectrum_data": ("XRD/Raman/XPS peak extraction", "SpectrumData"),
            "compare_structures": ("Multi-image quantitative comparison", "StructureComparison"),
            "validate_figure_caption": ("Figure-caption consistency check", "CaptionValidation"),
            "detect_anomaly": ("Anomaly detection + severity ranking", "AnomalyReport"),
            "extract_reaction": ("Reaction scheme extraction", "ReactionData"),
            "analyze_microscopy": ("Morphology, particle size, scale bar", "MicroscopyAnalysis"),
            "molecular_structure": ("SMILES, functional groups, stereocenters", "MolecularStructureData"),
            "property_prediction": ("Full pipeline: retrieval + encode + predict + MCTS", "PropertyPredictionResult"),
        }

        for name in skill_names:
            desc, output = skill_info.get(name, ("—", "—"))
            with st.expander(f"**{name}**"):
                st.markdown(f"**Description:** {desc}")
                st.markdown(f"**Output model:** `{output}`")
                skill_obj = DEFAULT_REGISTRY.get(name)
                try:
                    prompt = skill_obj.build_prompt()
                    st.markdown("**Default prompt:**")
                    st.code(prompt[:500] + ("..." if len(prompt) > 500 else ""), language=None)
                except (NotImplementedError, TypeError):
                    st.caption("(No default prompt — requires kwargs)")

    # ── system info ───────────────────────────────────────────────────────
    with bench_tab3:
        st.subheader("System Info")

        info_rows = []
        info_rows.append(("Python", sys.version.split()[0]))

        try:
            import torch
            info_rows.append(("PyTorch", torch.__version__))
            info_rows.append(("CUDA available", str(torch.cuda.is_available())))
            if torch.cuda.is_available():
                info_rows.append(("GPU", torch.cuda.get_device_name(0)))
        except ImportError:
            info_rows.append(("PyTorch", "not installed"))

        try:
            from rdkit import __version__ as rdkit_ver
            info_rows.append(("RDKit", rdkit_ver))
        except ImportError:
            info_rows.append(("RDKit", "not installed"))

        try:
            import spglib
            info_rows.append(("spglib", spglib.__version__))
        except (ImportError, AttributeError):
            info_rows.append(("spglib", "installed (version unknown)"))

        try:
            import chromadb
            info_rows.append(("ChromaDB", chromadb.__version__))
        except ImportError:
            info_rows.append(("ChromaDB", "not installed"))

        try:
            import mace
            info_rows.append(("MACE", "installed"))
        except ImportError:
            info_rows.append(("MACE-MP-0", "not installed (optional GPU tier)"))

        try:
            import unimol_tools
            info_rows.append(("Uni-Mol2", "installed"))
        except ImportError:
            info_rows.append(("Uni-Mol2", "not installed (optional)"))

        try:
            import anthropic
            info_rows.append(("Anthropic SDK", anthropic.__version__))
        except ImportError:
            info_rows.append(("Anthropic SDK", "not installed"))

        info_rows.append(("Streamlit", st.__version__))

        st.dataframe(pd.DataFrame(info_rows, columns=["Component", "Version / Status"]),
                     use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Architecture**")
        st.code("""
ChemVisionAgent Pipeline
========================

Image Input ──► VLM Perception (Claude / LLaVA)
                    │
                    ▼
              Skill Registry (9 skills)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
    Retrieval   Molecular   Physics
    (PubChem)   (RDKit)     (spglib)
         │          │          │
         └──────────┼──────────┘
                    ▼
            Property Prediction
                    │
                    ▼
           Pareto MCTS Optimisation
                    │
                    ▼
             Structured Report
        """, language=None)
