"""Lightweight paper figure loader for the Streamlit demo.

Fetches a PDF from an arXiv ID or DOI, extracts all figures as PIL images,
and returns them with their captions — without the full QA-generation pipeline.

This is intentionally thin: no Haiku calls, no ImageRecord construction.
It is designed for interactive use where the user wants to inspect and select
figures before running the agent.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image as PILImage


@dataclass
class PaperFigure:
    """A single figure extracted from a paper PDF."""

    figure_index: int          # 1-based
    page_num: int              # 0-based page in the PDF
    image: PILImage.Image
    caption: str = ""
    image_path: str = ""       # set after saving to disk


def fetch_figures(
    identifier: str,
    output_dir: str | Path,
    max_figures: int = 12,
    min_px: int = 100,
) -> list[PaperFigure]:
    """Download a paper and extract its figures.

    Parameters
    ----------
    identifier:
        arXiv ID (``"2301.00001"``, ``"arxiv:2301.00001"``) or
        DOI string (``"10.1038/..."``) of an open-access paper.
    output_dir:
        Directory where extracted PNG files are saved.
    max_figures:
        Maximum number of figures to return.
    min_px:
        Minimum width AND height in pixels; smaller images are skipped
        (they are usually logos, icons, or equation fragments).

    Returns
    -------
    list[PaperFigure]
        Figures in page order, each with its PIL image and caption.

    Raises
    ------
    ValueError
        When the identifier cannot be resolved to a PDF URL.
    RuntimeError
        When the PDF download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_url = _resolve_url(identifier)
    if pdf_url is None:
        raise ValueError(
            f"Cannot resolve {identifier!r} to a PDF URL. "
            "Use an arXiv ID (e.g. '2301.00001') or an open-access DOI."
        )

    pdf_bytes = _download(pdf_url)
    captions = _parse_captions(_extract_full_text(pdf_bytes))
    figures = _extract_figures(pdf_bytes, captions, max_figures, min_px)

    # Save images to disk so the agent can open them by path
    safe = re.sub(r"[^\w\-.]", "_", identifier)
    for fig in figures:
        dest = output_dir / f"{safe}_fig{fig.figure_index}.png"
        fig.image.save(dest, format="PNG")
        fig.image_path = str(dest)

    return figures


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ARXIV_PDF = "https://arxiv.org/pdf/{}"
_UNPAYWALL = "https://api.unpaywall.org/v2/{}?email=chemvision@example.com"
_HEADERS = {"User-Agent": "ChemVisionAgent/0.3 (scientific-research)"}


def _resolve_url(ident: str) -> str | None:
    ident = ident.strip()
    m = re.match(r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$", ident, re.IGNORECASE)
    if m:
        return _ARXIV_PDF.format(m.group(1))
    if ident.startswith("10."):
        return _resolve_doi(ident)
    return None


def _resolve_doi(doi: str) -> str | None:
    try:
        r = requests.get(_UNPAYWALL.format(doi), timeout=15, headers=_HEADERS)
        r.raise_for_status()
        data = r.json()
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception:
        pass
    return None


def _download(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=60, headers=_HEADERS)
        r.raise_for_status()
        return r.content
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download PDF from {url}: {exc}") from exc


def _extract_full_text(pdf_bytes: bytes) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("pypdf is required: pip install pypdf") from exc
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_captions(text: str) -> dict[int, str]:
    text = re.sub(r"\s+", " ", text)
    pattern = re.compile(
        r"(?:Fig(?:ure)?\.?\s*|FIG\.?\s*)(\d{1,3})[.\s|:]\s*(.*?)"
        r"(?=(?:Fig(?:ure)?\.?\s*|FIG\.?\s*)\d{1,3}[.\s|:]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    out: dict[int, str] = {}
    for m in pattern.finditer(text):
        n = int(m.group(1))
        if n not in out:
            out[n] = m.group(2).strip()[:800]
    return out


def _extract_figures(
    pdf_bytes: bytes,
    captions: dict[int, str],
    max_figures: int,
    min_px: int,
) -> list[PaperFigure]:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("pypdf is required: pip install pypdf") from exc

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    figures: list[PaperFigure] = []

    for page_num, page in enumerate(reader.pages):
        if len(figures) >= max_figures:
            break
        try:
            page_images = page.images
        except Exception:
            continue

        for img_obj in page_images:
            if len(figures) >= max_figures:
                break
            try:
                pil = PILImage.open(io.BytesIO(img_obj.data)).convert("RGB")
                w, h = pil.size
                if w < min_px or h < min_px:
                    continue
                idx = len(figures) + 1
                figures.append(
                    PaperFigure(
                        figure_index=idx,
                        page_num=page_num,
                        image=pil,
                        caption=captions.get(idx, ""),
                    )
                )
            except Exception:
                continue

    return figures
