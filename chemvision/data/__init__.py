"""Dataset construction pipeline for ChemVision Agent.

Public API
----------
DatasetBuilder      -- orchestrates raw-image → annotation → split pipeline
ImageRecord         -- Pydantic schema for a single annotated sample
DatasetConfig       -- top-level configuration for a build run
SyntheticConfig     -- config for the VASP / LAMMPS synthetic generator
ScraperConfig       -- config for the arXiv / DOI literature scraper
SyntheticGenerator  -- renders simulation files → images + QA pairs
LiteratureScraper   -- fetches papers → figures + captions → QA pairs
ParsedStructure     -- parsed crystallographic structure (ASE Atoms wrapper)
FigureCaption       -- extracted figure image + caption from a PDF
print_stats         -- print a summary of a saved HuggingFace dataset
"""

from chemvision.data.builder import DatasetBuilder
from chemvision.data.data_stats import print_stats
from chemvision.data.schema import (
    DatasetConfig,
    ImageDomain,
    ImageRecord,
    ScraperConfig,
    SyntheticConfig,
)
from chemvision.data.scraper import FigureCaption, LiteratureScraper
from chemvision.data.synthetic import ParsedStructure, SyntheticGenerator

__all__ = [
    "DatasetBuilder",
    "DatasetConfig",
    "FigureCaption",
    "ImageDomain",
    "ImageRecord",
    "LiteratureScraper",
    "ParsedStructure",
    "ScraperConfig",
    "SyntheticConfig",
    "SyntheticGenerator",
    "print_stats",
]
