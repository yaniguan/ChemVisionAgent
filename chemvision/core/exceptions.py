"""Custom exception hierarchy for ChemVisionAgent.

All domain-specific exceptions inherit from :class:`ChemVisionError` so
callers can catch a single base class when they want a blanket handler,
while still being able to discriminate by sub-type for finer control.
"""


class ChemVisionError(Exception):
    """Base exception for all ChemVisionAgent errors."""


class MoleculeParsingError(ChemVisionError):
    """Raised when a SMILES / SELFIES string cannot be parsed into a molecule."""


class ModelInferenceError(ChemVisionError):
    """Raised when a model forward pass fails (OOM, bad input shape, etc.)."""


class DataLoadError(ChemVisionError):
    """Raised when data files or images cannot be read from disk or network."""


class ConfigurationError(ChemVisionError):
    """Raised when configuration values are invalid or missing."""


class EvaluationError(ChemVisionError):
    """Raised when a metric computation or evaluation step fails."""
