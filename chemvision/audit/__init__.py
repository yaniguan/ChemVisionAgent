"""Capability auditing framework for ChemVision Agent.

Public API
----------
AuditRunner          -- runs a suite of skill probes against a model
AuditReport          -- structured result with per-skill scores
AuditConfig          -- configuration for an audit run
CapabilityMatrix     -- 2-axis accuracy matrix (task_type × difficulty)
MatrixConfig         -- configuration for CapabilityMatrix
DegradationTester    -- binary-search robustness envelope under image degradations
DegradationConfig    -- configuration for DegradationTester
ReliabilityEnvelope  -- serialisable robustness envelope output
AuditReportGenerator -- assembles markdown report with embedded heatmap
"""

from chemvision.audit.config import AuditConfig
from chemvision.audit.degradation import (
    DegradationConfig,
    DegradationResult,
    DegradationTester,
    ReliabilityEnvelope,
)
from chemvision.audit.matrix import CapabilityMatrix, MatrixConfig, TaskType, Difficulty
from chemvision.audit.report import AuditReport
from chemvision.audit.report_generator import AuditReportGenerator
from chemvision.audit.runner import AuditRunner

__all__ = [
    "AuditConfig",
    "AuditReport",
    "AuditRunner",
    "CapabilityMatrix",
    "MatrixConfig",
    "TaskType",
    "Difficulty",
    "DegradationTester",
    "DegradationConfig",
    "DegradationResult",
    "ReliabilityEnvelope",
    "AuditReportGenerator",
]
