"""Generate ChemVisionAgent pipeline architecture figure (no-emoji version)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties

# ── colour palette ─────────────────────────────────────────────────────────
C = {
    "bg":       "#0f1117",
    "panel":    "#1a1d27",
    "user":     "#3b82f6",
    "brain":    "#8b5cf6",
    "model":    "#f59e0b",
    "skill":    "#10b981",
    "report":   "#ef4444",
    "audit":    "#06b6d4",
    "finetune": "#ec4899",
    "arrow":    "#94a3b8",
    "text":     "#f1f5f9",
    "subtext":  "#94a3b8",
    "badge":    "#1e293b",
    "dim":      "#334155",
}

fig = plt.figure(figsize=(24, 14), facecolor=C["bg"])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 24)
ax.set_ylim(0, 14)
ax.axis("off")
ax.set_facecolor(C["bg"])

# ── helpers ────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc, ec, lw=1.8, r=0.3, alpha=0.95):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       alpha=alpha, zorder=3)
    ax.add_patch(p)

def txt(x, y, s, sz=11, col=C["text"], w="bold", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=sz, color=col, fontweight=w, fontstyle=style,
            ha=ha, va=va, zorder=6, fontfamily="monospace")

def stxt(x, y, s, sz=8.5, col=C["subtext"], ha="center"):
    ax.text(x, y, s, fontsize=sz, color=col, fontweight="normal",
            ha=ha, va="center", zorder=6)

def arr(x0, y0, x1, y1, col=C["arrow"], lw=1.8, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def darr(x0, y0, x1, y1, col=C["arrow"], lw=1.6, rad=0.0):
    """Double-headed arrow."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<|-|>", color=col, lw=lw,
                                mutation_scale=12,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)

def htag(x, y, label, col):
    """Small coloured tag."""
    ax.text(x, y, label, fontsize=7.5, color="white", fontweight="bold",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=col,
                      edgecolor="none", alpha=0.95))


# ══════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════
txt(12, 13.45, "ChemVision Agent  —  Pipeline Architecture", sz=20, col=C["text"])
stxt(12, 12.95, "ReAct-based multimodal scientific image reasoning platform  |  v0.2", sz=11)
ax.axhline(12.65, color=C["dim"], lw=0.8, xmin=0.015, xmax=0.985, zorder=2)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 1  —  INPUT
# ══════════════════════════════════════════════════════════════════════════
rbox(0.5, 10.5, 4.2, 2.0, C["panel"], C["user"])
txt(2.6, 12.15,  "USER INPUT",          sz=12, col=C["user"])
stxt(2.6, 11.73, "question  +  image_paths[ ]", sz=9.5)
for i, (lbl, col) in enumerate([
    ("FastAPI  POST /analyze", C["user"]),
    ("Streamlit demo/app.py",  C["user"]),
    ("Python API  agent.run()", C["user"]),
]):
    rbox(0.6 + i*1.37, 10.56, 1.28, 0.46, C["badge"], col, lw=1.0, r=0.2)
    stxt(0.6 + i*1.37 + 0.64, 10.79, lbl, sz=7.5, col=col)

arr(4.7, 11.5, 5.7, 11.5)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 1  —  CLAUDE PLANNER
# ══════════════════════════════════════════════════════════════════════════
rbox(5.7, 10.15, 5.0, 2.35, C["panel"], C["brain"])
txt(8.2, 12.17, "CLAUDE PLANNER",        sz=12, col=C["brain"])
stxt(8.2, 11.75, "claude-sonnet-4-20250514", sz=9.5)
stxt(8.2, 11.4,  "Anthropic tool-use API  (AgentPlanner)", sz=8.5)

# ReAct steps row
react = [("Thought", "#a78bfa"), ("Action", "#34d399"), ("Observe", "#fbbf24")]
for i, (lbl, col) in enumerate(react):
    rbox(5.8 + i*1.58, 10.22, 1.45, 0.68, C["badge"], col, lw=1.3, r=0.2)
    txt(5.8 + i*1.58 + 0.72, 10.56, lbl, sz=9, col=col)

arr(10.7, 11.32, 11.7, 11.32)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 1  —  VISION MODEL
# ══════════════════════════════════════════════════════════════════════════
rbox(11.7, 10.15, 4.5, 2.35, C["panel"], C["model"])
txt(13.95, 12.17, "VISION MODEL",           sz=12, col=C["model"])
stxt(13.95, 11.75, "generate(image, prompt) -> str", sz=9.5)
for i, (lbl, col) in enumerate([
    ("Anthropic (default)", C["model"]),
    ("LLaVA-1.6  GPU",      "#fbbf24"),
    ("Qwen-VL  GPU",        "#fbbf24"),
]):
    rbox(11.8 + i*1.43, 10.22, 1.33, 0.68, C["badge"], col, lw=1.0, r=0.2)
    stxt(11.8 + i*1.43 + 0.66, 10.56, lbl, sz=7.5, col=col)

# bidirectional arrow planner <-> model
darr(10.7, 11.0, 11.7, 11.0, col=C["dim"], lw=1.2)
stxt(11.2, 10.78, "tool\ncalls", sz=7.5, col=C["dim"])


# ══════════════════════════════════════════════════════════════════════════
#  VERTICAL ARROWS → SKILLS
# ══════════════════════════════════════════════════════════════════════════
arr(8.2,  10.15, 8.2,  9.35)
arr(13.95, 10.15, 13.95, 9.35)
# merge lines (decorative)
ax.plot([8.2, 11.1], [9.35, 9.35], color=C["arrow"], lw=1.8, zorder=4)
ax.plot([13.95, 11.1], [9.35, 9.35], color=C["arrow"], lw=1.8, zorder=4)
arr(11.1, 9.35, 11.1, 9.15)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 2  —  SKILL REGISTRY
# ══════════════════════════════════════════════════════════════════════════
rbox(0.4, 7.55, 23.0, 1.5, C["panel"], C["skill"])
txt(11.9, 8.83, "SKILL REGISTRY  (DEFAULT_REGISTRY)",  sz=12, col=C["skill"])

skills = [
    ("analyze_structure",         "Structure Analysis\nlattice · symmetry · defects"),
    ("extract_spectrum_data",     "Spectrum Extraction\nXRD · Raman · XPS peaks"),
    ("compare_structures",        "Structure Comparison\ndiff regions · quantitative"),
    ("validate_figure_caption",   "Caption Validation\nconsistency score"),
    ("detect_anomaly",            "Anomaly Detection\ntype · severity · location"),
]
sw, sg = 4.35, 0.18
for i, (name, desc) in enumerate(skills):
    bx = 0.5 + i*(sw+sg)
    rbox(bx, 7.62, sw, 0.96, C["badge"], C["skill"], lw=1.2, r=0.22)
    txt(bx+sw/2, 8.24, name, sz=8.5, col="#6ee7b7")
    stxt(bx+sw/2, 7.93, desc, sz=7.5, col=C["subtext"])

arr(11.9, 7.55, 11.9, 6.85)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 3  —  PYDANTIC OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
rbox(0.4, 5.65, 23.0, 1.1, C["panel"], C["dim"])
txt(11.9, 6.52, "TYPED PYDANTIC OUTPUT MODELS  (SkillResult subclasses)", sz=11, col="#64748b")

outputs = [
    ("StructureAnalysis",    "LatticeParams\nDefectLocation[ ]"),
    ("SpectrumData",         "Peak[ ]\nspectrum_type"),
    ("StructureComparison",  "DiffRegion[ ]\nQuantitativeChange[ ]"),
    ("CaptionValidation",    "consistency_score\ncontradictions[ ]"),
    ("AnomalyReport",        "Anomaly[ ]\nseverity: none/low/med/high"),
]
ow, og = 4.35, 0.18
for i, (name, fields) in enumerate(outputs):
    bx = 0.5 + i*(ow+og)
    rbox(bx, 5.72, ow, 0.82, C["badge"], C["dim"], lw=1.0, r=0.18)
    txt(bx+ow/2, 6.22, name, sz=8.5, col="#94a3b8")
    stxt(bx+ow/2, 5.92, fields, sz=7.5, col="#64748b")

arr(11.9, 5.65, 11.9, 4.95)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 4  —  ANALYSIS REPORT
# ══════════════════════════════════════════════════════════════════════════
rbox(1.2, 3.75, 21.5, 1.1, C["panel"], C["report"])
txt(11.95, 4.62, "ANALYSIS REPORT  (AnalysisReport.build())", sz=12, col=C["report"])

report_fields = [
    ("final_answer",           C["report"]),
    ("structured_data {}",     C["report"]),
    ("tool_logs[ ]",           C["report"]),
    ("min_confidence",         "#fca5a5"),
    ("low_confidence_flag",    "#ef4444"),
    ("num_steps",              "#fca5a5"),
    ("reasoning trace",        "#fca5a5"),
]
rw = 2.95
for i, (f, col) in enumerate(report_fields):
    bx = 1.3 + i*(rw+0.14)
    rbox(bx, 3.82, rw, 0.72, C["badge"], col, lw=0.9, r=0.18)
    txt(bx+rw/2, 4.18, f, sz=8.5, col=col)


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE PROPAGATION callout
# ══════════════════════════════════════════════════════════════════════════
rbox(16.8, 5.7, 6.8, 1.55, "#1a0a1a", C["report"], lw=2.0, r=0.3)
txt(20.2, 6.98, "CONFIDENCE PROPAGATION", sz=9.5, col=C["report"])
stxt(20.2, 6.62, "each ToolCallLog records confidence (0-1)", sz=8.5)
stxt(20.2, 6.3,  "any skill < threshold (0.75)", sz=8.5)
stxt(20.2, 5.98, "=> low_confidence_flag = True on report", sz=8.5, col="#ef4444")
arr(20.2, 5.7, 17.5, 4.85, col=C["report"], lw=1.5)


# ══════════════════════════════════════════════════════════════════════════
#  LAYER 5  —  AUDIT  +  FINE-TUNE
# ══════════════════════════════════════════════════════════════════════════
arr(8.0,  3.75, 5.0,  2.95, rad=-0.1)
arr(15.8, 3.75, 19.0, 2.95, rad=0.1)

# Audit box
rbox(0.4, 1.4, 9.2, 1.45, C["panel"], C["audit"])
txt(5.0, 2.62, "AUDIT FRAMEWORK", sz=12, col=C["audit"])
audit_items = [
    ("CapabilityMatrix",     "5 tasks x 3 difficulties\nheatmap export"),
    ("DegradationTester",    "5 degradation types\nbinary-search robustness"),
    ("AuditReportGenerator", "markdown + base64\nheatmap embedding"),
]
aw = 2.9
for i, (name, desc) in enumerate(audit_items):
    bx = 0.5 + i*(aw+0.14)
    rbox(bx, 1.47, aw, 0.84, C["badge"], C["audit"], lw=1.0, r=0.2)
    txt(bx+aw/2, 1.99, name, sz=8.5, col="#67e8f9")
    stxt(bx+aw/2, 1.67, desc, sz=7.5)

# Fine-tune box
rbox(14.4, 1.4, 9.2, 1.45, C["panel"], C["finetune"])
txt(19.0, 2.62, "FINE-TUNING PIPELINE", sz=12, col=C["finetune"])
ft_items = [
    ("DatasetBuilder",           "LiteratureScraper\nSyntheticGenerator"),
    ("PeftFineTuner  (LoRA)",    "configs/lora_train.yaml\nscripts/train_lora.py"),
    ("DynResolutionEncoder",     "gradient saliency\npatch budget allocation"),
]
for i, (name, desc) in enumerate(ft_items):
    bx = 14.5 + i*(aw+0.14)
    rbox(bx, 1.47, aw, 0.84, C["badge"], C["finetune"], lw=1.0, r=0.2)
    txt(bx+aw/2, 1.99, name, sz=8.5, col="#f9a8d4")
    stxt(bx+aw/2, 1.67, desc, sz=7.5)

# feedback arrow: fine-tune -> vision model
ax.annotate("", xy=(16.2, 10.15), xytext=(23.0, 2.15),
            arrowprops=dict(arrowstyle="-|>", color=C["finetune"], lw=1.5,
                            mutation_scale=13,
                            connectionstyle="arc3,rad=-0.22"), zorder=5)
stxt(22.35, 6.3, "fine-tuned\nweights", sz=8.5, col=C["finetune"])

# CLI tag
rbox(9.75, 1.47, 4.2, 0.84, C["badge"], C["dim"], lw=1.0, r=0.2)
txt(11.85, 1.99, "CLI  /  REST API", sz=9, col=C["subtext"])
stxt(11.85, 1.67, "chemvision serve  |  uvicorn  |  python -m chemvision.audit.run", sz=7.5)


# ══════════════════════════════════════════════════════════════════════════
#  LEGEND
# ══════════════════════════════════════════════════════════════════════════
legend_x, legend_y = 0.55, 5.3
items = [
    (C["user"],     "User Interface"),
    (C["brain"],    "Claude Planner  (ReAct)"),
    (C["model"],    "Vision Model Backend"),
    (C["skill"],    "Skill Modules"),
    (C["report"],   "Report / Output"),
    (C["audit"],    "Audit Framework"),
    (C["finetune"], "Fine-tuning Pipeline"),
]
rbox(legend_x-0.15, legend_y-len(items)*0.31-0.12, 3.7, len(items)*0.31+0.5,
     C["panel"], C["dim"], lw=1.0)
txt(legend_x+1.7, legend_y+0.1, "LEGEND", sz=9, col=C["subtext"], w="normal")
for i, (col, label) in enumerate(items):
    y = legend_y - 0.31*(i+1) + 0.15
    ax.add_patch(plt.Rectangle((legend_x, y-0.07), 0.32, 0.22,
                                color=col, zorder=7))
    stxt(legend_x + 0.5, y+0.04, label, sz=8.5, ha="left")


# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
ax.axhline(1.1, color=C["dim"], lw=0.6, xmin=0.015, xmax=0.985, zorder=2)
stxt(12, 0.65,
     "ChemVisionAgent v0.2  |  Python 3.11  |  Pydantic v2  |  "
     "Anthropic SDK  |  FastAPI  |  Streamlit  |  HuggingFace Transformers  |  PEFT / LoRA",
     sz=8.5, col="#475569")

# ── save ──────────────────────────────────────────────────────────────────
os.makedirs("docs", exist_ok=True)
out = "docs/pipeline.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C["bg"])
print(f"saved -> {out}")
plt.close(fig)
