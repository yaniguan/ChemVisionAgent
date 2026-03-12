"""ChemVision Agent — Streamlit Demo

Upload one or more scientific images, type a question, and watch the agent
reason through which vision skills to call.  The reasoning trace and final
structured report are displayed side-by-side.

Run
---
    streamlit run demo/app.py

Requirements
------------
    pip install streamlit anthropic pillow matplotlib
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChemVision Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_agent(api_key: str, model: str, threshold: float, max_steps: int):
    """Construct and return a ChemVisionAgent, cached by config."""
    from chemvision.agent.agent import ChemVisionAgent
    from chemvision.agent.config import AgentConfig

    cfg = AgentConfig(
        anthropic_api_key=api_key,
        planning_model=model,
        confidence_threshold=threshold,
        max_steps=max_steps,
        verbose=False,
    )
    return ChemVisionAgent(cfg)


@st.cache_resource(show_spinner=False)
def _get_agent(api_key: str, model: str, threshold: float, max_steps: int):
    return _load_agent(api_key, model, threshold, max_steps)


def _step_icon(step_type: str) -> str:
    return {"thought": "💭", "action": "⚡", "observation": "👁️", "final_answer": "✅"}.get(
        step_type, "•"
    )


def _confidence_color(conf: float | None) -> str:
    if conf is None:
        return "gray"
    if conf >= 0.85:
        return "green"
    if conf >= 0.70:
        return "orange"
    return "red"


def _render_trace(steps: list[dict[str, Any]]) -> None:
    for step in steps:
        icon = _step_icon(step["step_type"])
        label = step["step_type"].replace("_", " ").title()
        skill = f" · `{step['skill_name']}`" if step.get("skill_name") else ""

        with st.expander(f"{icon} Step {step['step_index']} — {label}{skill}", expanded=False):
            content = step["content"]
            # Try to pretty-print JSON content (action steps)
            try:
                parsed = json.loads(content)
                st.json(parsed)
            except (json.JSONDecodeError, TypeError):
                st.markdown(f"```\n{content}\n```")


def _render_tool_log(log: dict[str, Any], threshold: float) -> None:
    conf = log.get("confidence")
    color = _confidence_color(conf)
    conf_str = f"{conf:.2f}" if conf is not None else "—"
    low = log.get("low_confidence", False)
    warning = " ⚠️" if low else ""

    with st.expander(
        f"🔧 `{log['skill_name']}`  — confidence: :{color}[**{conf_str}**]{warning}",
        expanded=low,
    ):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Inputs**")
            st.json(log.get("inputs", {}))
        with col2:
            st.markdown("**Output summary**")
            st.text(log.get("output_summary", ""))


def _save_uploads(uploaded_files) -> list[str]:
    """Save uploaded files to /tmp and return their paths."""
    paths = []
    for f in uploaded_files:
        dest = Path("/tmp") / f.name
        dest.write_bytes(f.read())
        paths.append(str(dest))
    return paths


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 ChemVision")
    st.caption("Scientific image reasoning agent")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your Anthropic API key. Not stored beyond this session.",
    )

    st.subheader("Agent settings")
    planning_model = st.selectbox(
        "Planning model",
        ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
        index=0,
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Skill outputs below this confidence are flagged in the report.",
    )
    max_steps = st.number_input(
        "Max ReAct steps",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )

    st.divider()
    st.subheader("Quick examples")
    example_questions = [
        "What crystal phase is dominant in this XRD pattern?",
        "Identify any defects or anomalies in this SEM image.",
        "Extract all peak positions from this Raman spectrum.",
        "Compare the grain morphology across all uploaded images.",
        "Does the caption accurately describe the figure?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["question_prefill"] = q

    st.divider()
    st.caption("ChemVision Agent v0.2 · Powered by Claude")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("ChemVision Agent")
st.markdown(
    "Upload scientific images, ask a question, and let the agent reason step-by-step "
    "using composable vision skills."
)

# Image upload
st.subheader("1 · Upload images")
uploaded_files = st.file_uploader(
    "Drop image files here (PNG, JPG, TIFF)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded_files:
    cols = st.columns(min(len(uploaded_files), 4))
    for col, f in zip(cols, uploaded_files):
        with col:
            img = Image.open(io.BytesIO(f.read()))
            f.seek(0)  # reset for later save
            st.image(img, caption=f.name, use_container_width=True)

# Question input
st.subheader("2 · Ask a question")
prefill = st.session_state.pop("question_prefill", "")
question = st.text_area(
    "Question",
    value=prefill,
    placeholder="e.g. What phase transition occurs between these XRD patterns?",
    height=90,
    label_visibility="collapsed",
)

# Run button
run_disabled = not (api_key and uploaded_files and question.strip())
run_hint = "" if not run_disabled else " ← add API key, images, and a question"

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("▶ Analyze", type="primary", disabled=run_disabled)
with col_hint:
    st.caption(run_hint)

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------

if run_clicked:
    image_paths = _save_uploads(uploaded_files)
    agent = _get_agent(api_key, planning_model, confidence_threshold, int(max_steps))

    result_placeholder = st.empty()
    result_placeholder.info("Agent is running…")

    t0 = time.perf_counter()
    with st.spinner("Reasoning…"):
        try:
            from chemvision.agent.report import AnalysisReport

            report: AnalysisReport = agent.run(
                question=question.strip(),
                image_paths=image_paths,
            )
            elapsed = time.perf_counter() - t0
            st.session_state["last_report"] = report
            st.session_state["last_elapsed"] = elapsed
            result_placeholder.empty()
        except Exception as exc:
            result_placeholder.error(f"Agent error: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "last_report" in st.session_state:
    report: AnalysisReport = st.session_state["last_report"]
    elapsed: float = st.session_state["last_elapsed"]

    st.divider()
    st.subheader("3 · Results")

    # --- Summary metrics ---------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steps", report.num_steps)
    m2.metric("Skill calls", len(report.tool_logs))
    conf_label = (
        f"{report.min_intermediate_confidence:.2f}"
        if report.min_intermediate_confidence is not None
        else "—"
    )
    m3.metric("Min confidence", conf_label, delta=None)
    m4.metric("Latency", f"{elapsed:.1f}s")

    if report.low_confidence_flag:
        st.warning(
            "⚠️ One or more intermediate skill calls had confidence below the threshold "
            f"({confidence_threshold:.2f}). Treat conclusions with caution."
        )
    else:
        st.success("All confidence scores are above the threshold.")

    # --- Two-column layout -------------------------------------------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Reasoning trace")
        st.caption(f"{report.num_steps} steps")
        trace_steps: list[dict[str, Any]] = []
        # Rebuild trace steps from tool logs for display
        # (AgentTrace not persisted on report — reconstruct from tool_logs)
        for i, log in enumerate(report.tool_logs, start=1):
            trace_steps.append({
                "step_index": i,
                "step_type": "action",
                "skill_name": log.skill_name,
                "content": json.dumps({"inputs": log.inputs}, indent=2),
            })
            trace_steps.append({
                "step_index": i,
                "step_type": "observation",
                "skill_name": log.skill_name,
                "content": log.output_summary,
            })

        if trace_steps:
            _render_trace(trace_steps)
        else:
            st.info("No intermediate steps recorded.")

    with right:
        st.markdown("### Final answer")
        # Try to display as structured JSON first
        if report.structured_data:
            st.json(report.structured_data)
        else:
            st.markdown(report.final_answer)

        st.markdown("### Tool call log")
        st.caption(f"{len(report.tool_logs)} skill invocation(s)")
        for log in report.tool_logs:
            _render_tool_log(log.model_dump(), confidence_threshold)

        if report.structured_data:
            st.markdown("### Structured data")
            # Render grain_size_vs_temperature as a chart if present
            gsvt = report.structured_data.get("grain_size_vs_temperature")
            if gsvt and isinstance(gsvt, list):
                import pandas as pd  # type: ignore[import]

                df = pd.DataFrame(gsvt)
                if "temperature_c" in df.columns and "grain_size_nm" in df.columns:
                    st.markdown("**Grain size vs. temperature**")
                    st.line_chart(
                        df.set_index("temperature_c")["grain_size_nm"],
                        x_label="Temperature (°C)",
                        y_label="Grain size (nm)",
                    )

            ptt = report.structured_data.get("phase_transition_temperature")
            if ptt is not None:
                unit = report.structured_data.get("phase_transition_temperature_unit", "°C")
                st.info(f"**Phase transition temperature: {ptt} {unit}**")

    # --- Raw report JSON (collapsible) ------------------------------------
    with st.expander("Raw report JSON"):
        st.json(report.to_dict())
