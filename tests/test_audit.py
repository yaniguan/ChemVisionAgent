"""Tests for chemvision.audit module."""

from chemvision.audit.report import AuditReport, SkillScore


def test_audit_report_summary() -> None:
    report = AuditReport(
        model_name="test-model",
        skill_scores=[
            SkillScore(skill_name="spectrum_reading", accuracy=0.85, num_samples=100),
            SkillScore(skill_name="molecular_structure", accuracy=0.70, num_samples=80),
        ],
        overall_accuracy=0.775,
    )
    summary = report.summary()
    assert "test-model" in summary
    assert "77.50%" in summary


def test_audit_report_empty() -> None:
    report = AuditReport(model_name="empty-model")
    assert report.skill_scores == []
    assert report.overall_accuracy == 0.0
