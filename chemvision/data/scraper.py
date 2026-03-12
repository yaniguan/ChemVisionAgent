"""Literature scraper: DOI / arXiv ID → PDF → figures + captions → structured QA pairs.

Pipeline
--------
1. Fetch the PDF from arXiv (direct URL) or via Unpaywall for DOIs.
2. Extract embedded images (pypdf) and caption text (regex) from the PDF.
3. Call Claude Haiku to parse each caption into 2–3 structured QA pairs,
   optionally grounding the call with the figure image for visual context.
4. Return a list of :class:`~chemvision.data.schema.ImageRecord` objects.

Requires
--------
pypdf>=4.2, anthropic>=0.40, requests>=2.32, pillow>=10
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from PIL import Image as PILImage

from chemvision.data.schema import ImageDomain, ImageRecord


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FigureCaption:
    """A figure image paired with its caption text extracted from a paper."""

    figure_num: int | str
    caption: str
    image_data: bytes  # raw PNG bytes (converted from whatever the PDF contained)
    page_num: int
    source_id: str  # DOI or arXiv ID


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class LiteratureScraper:
    """Scrape scientific figures and generate QA pairs via Claude Haiku.

    Example
    -------
    >>> scraper = LiteratureScraper(output_dir=Path("data/raw/literature"))
    >>> records = scraper.scrape(["2301.00001", "10.1038/s41586-023-06094-5"])
    """

    _ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
    _UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email=chemvision@example.com"
    _HAIKU_MODEL = "claude-haiku-4-5"

    def __init__(
        self,
        output_dir: Path,
        api_key: str | None = None,
        max_figures_per_paper: int = 10,
        request_delay: float = 1.0,
        include_image_in_haiku_call: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key  # Falls back to ANTHROPIC_API_KEY env var
        self.max_figures_per_paper = max_figures_per_paper
        self.request_delay = request_delay
        self.include_image_in_haiku_call = include_image_in_haiku_call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, identifiers: list[str]) -> list[ImageRecord]:
        """Scrape a list of DOIs or arXiv IDs and return :class:`ImageRecord` objects.

        Errors for individual papers are logged to stdout and skipped so the
        rest of the batch still completes.

        Parameters
        ----------
        identifiers:
            List of arXiv IDs (e.g. ``"2301.00001"``) or DOI strings
            (e.g. ``"10.1038/s41586-023-00001-0"``).

        Returns
        -------
        list[ImageRecord]
        """
        all_records: list[ImageRecord] = []
        for ident in identifiers:
            try:
                records = self._process_identifier(ident)
                all_records.extend(records)
            except Exception as exc:
                print(f"[LiteratureScraper] Skipping {ident!r}: {exc}")
            time.sleep(self.request_delay)
        return all_records

    # ------------------------------------------------------------------
    # PDF fetching
    # ------------------------------------------------------------------

    def fetch_pdf(self, doi_or_arxiv: str) -> bytes | None:
        """Download a PDF for an arXiv ID or DOI.

        For arXiv IDs uses ``https://arxiv.org/pdf/{id}`` directly.
        For DOIs queries Unpaywall for an open-access PDF URL.

        Parameters
        ----------
        doi_or_arxiv:
            arXiv ID (``"2301.00001"`` / ``"arxiv:2301.00001"``) or
            DOI string (``"10.1038/..."``).

        Returns
        -------
        bytes or None
            Raw PDF bytes, or ``None`` when the download fails.
        """
        url = self._resolve_url(doi_or_arxiv)
        if url is None:
            print(f"[LiteratureScraper] Could not resolve URL for {doi_or_arxiv!r}")
            return None
        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "ChemVisionAgent/0.1 (scientific-research)"},
            )
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            print(f"[LiteratureScraper] Failed to fetch {url}: {exc}")
            return None

    def _resolve_url(self, ident: str) -> str | None:
        """Resolve an identifier string to a direct PDF download URL."""
        ident = ident.strip()

        # arXiv: matches "2301.00001", "2301.00001v2", "arxiv:2301.00001"
        arxiv_pattern = re.compile(
            r"^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$",
            re.IGNORECASE,
        )
        m = arxiv_pattern.match(ident)
        if m:
            return self._ARXIV_PDF_URL.format(arxiv_id=m.group(1))

        # DOI: starts with "10."
        if ident.startswith("10."):
            return self._resolve_doi(ident)

        return None

    def _resolve_doi(self, doi: str) -> str | None:
        """Query Unpaywall for an open-access PDF URL for a DOI."""
        url = self._UNPAYWALL_URL.format(doi=doi)
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "ChemVisionAgent/0.1 (scientific-research)"},
            )
            resp.raise_for_status()
            data = resp.json()
            # Prefer best_oa_location
            best = data.get("best_oa_location")
            if best and best.get("url_for_pdf"):
                return best["url_for_pdf"]
            # Fallback: scan all OA locations
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    return loc["url_for_pdf"]
        except Exception as exc:
            print(f"[LiteratureScraper] Unpaywall lookup failed for {doi}: {exc}")
        return None

    # ------------------------------------------------------------------
    # Figure extraction
    # ------------------------------------------------------------------

    def extract_figures(self, pdf_bytes: bytes, source_id: str) -> list[FigureCaption]:
        """Extract embedded figures and their captions from raw PDF bytes.

        Uses pypdf to pull images from each page (skipping thumbnails < 50 px)
        and regex to match ``Figure N.`` / ``Fig. N.`` captions in the text.

        Parameters
        ----------
        pdf_bytes:
            Raw bytes of the PDF document.
        source_id:
            Identifier stored verbatim in the returned :class:`FigureCaption` objects.

        Returns
        -------
        list[FigureCaption]
            At most ``self.max_figures_per_paper`` items, in page order.
        """
        try:
            import pypdf
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for figure extraction. Install with: pip install pypdf"
            ) from exc

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        figures: list[FigureCaption] = []

        # Build caption map from full document text (more reliable than per-page)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        caption_map = self.parse_captions(full_text)

        for page_num, page in enumerate(reader.pages):
            if len(figures) >= self.max_figures_per_paper:
                break
            try:
                page_images = page.images
            except Exception:
                continue

            for img in page_images:
                if len(figures) >= self.max_figures_per_paper:
                    break
                try:
                    pil_img = PILImage.open(io.BytesIO(img.data))
                    w, h = pil_img.size
                    if w < 50 or h < 50:  # skip tiny icons / logos
                        continue

                    buf = io.BytesIO()
                    pil_img.convert("RGB").save(buf, format="PNG")
                    png_bytes = buf.getvalue()

                    fig_num = len(figures) + 1
                    figures.append(
                        FigureCaption(
                            figure_num=fig_num,
                            caption=caption_map.get(fig_num, ""),
                            image_data=png_bytes,
                            page_num=page_num,
                            source_id=source_id,
                        )
                    )
                except Exception:
                    continue

        return figures

    @staticmethod
    def parse_captions(text: str) -> dict[int, str]:
        """Extract figure captions from document text using regex.

        Matches patterns such as::

            Figure 1. Caption text up to the next figure marker.
            Fig. 2. Caption text.
            FIG. 3 | Caption text.

        Parameters
        ----------
        text:
            Full document text (whitespace-normalised internally).

        Returns
        -------
        dict[int, str]
            Mapping ``{figure_number: caption_text}``.
        """
        # Collapse whitespace for more reliable matching
        text = re.sub(r"\s+", " ", text)

        pattern = re.compile(
            r"(?:Fig(?:ure)?\.?\s*|FIG\.?\s*)(\d{1,3})"  # "Figure N" / "Fig. N"
            r"[.\s|:]\s*"                                   # separator
            r"(.*?)"                                        # caption body (lazy)
            r"(?="                                          # lookahead for end
            r"(?:Fig(?:ure)?\.?\s*|FIG\.?\s*)\d{1,3}[.\s|:]"  # next figure
            r"|\Z"                                          # or end of string
            r")",
            re.IGNORECASE | re.DOTALL,
        )

        caption_map: dict[int, str] = {}
        for m in pattern.finditer(text):
            fig_num = int(m.group(1))
            caption_text = m.group(2).strip()[:1000]  # cap very long captions
            if caption_text and fig_num not in caption_map:
                caption_map[fig_num] = caption_text
        return caption_map

    # ------------------------------------------------------------------
    # QA generation via Claude Haiku
    # ------------------------------------------------------------------

    def parse_caption_with_haiku(
        self,
        caption: str,
        figure_num: int | str,
        image_bytes: bytes | None = None,
    ) -> list[dict[str, str]]:
        """Call Claude Haiku to parse a figure caption into structured QA pairs.

        Sends the caption (and optionally the figure image for visual grounding)
        to Claude Haiku and asks for 2–3 JSON-formatted QA pairs.

        Parameters
        ----------
        caption:
            Figure caption text from the paper.
        figure_num:
            Figure number, used in the prompt.
        image_bytes:
            Optional PNG bytes of the figure.  When supplied the image is
            included in the request for richer, visually-grounded QA.

        Returns
        -------
        list[dict[str, str]]
            Each dict contains ``"question"``, ``"answer"``, ``"difficulty"``.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK is required. Install with: pip install anthropic"
            ) from exc

        client = (
            anthropic.Anthropic(api_key=self._api_key)
            if self._api_key
            else anthropic.Anthropic()
        )

        system_prompt = (
            "You are an expert scientific image analyst specialising in chemistry "
            "and materials science. Given a figure caption from a peer-reviewed paper, "
            "generate structured question-answer pairs that test factual understanding "
            "of the figure. Focus on quantitative, specific, directly answerable questions."
        )

        user_content: list[dict[str, Any]] = []

        # Optionally attach the figure image for visual grounding
        if image_bytes is not None and self.include_image_in_haiku_call:
            b64_data = base64.standard_b64encode(image_bytes).decode()
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64_data,
                    },
                }
            )

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Figure {figure_num} caption:\n{caption}\n\n"
                    "Generate 2–3 question-answer pairs grounded in this caption.\n"
                    "Return ONLY valid JSON — no markdown fences, no extra text:\n"
                    "[\n"
                    '  {"question": "...", "answer": "...", "difficulty": "easy|medium|hard"},\n'
                    "  ...\n"
                    "]"
                ),
            }
        )

        try:
            response = client.messages.create(
                model=self._HAIKU_MODEL,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            raw_text = next(
                (b.text for b in response.content if b.type == "text"),
                "[]",
            )
            # Strip accidental markdown code fences
            raw_text = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE
            )
            pairs = json.loads(raw_text)
            if not isinstance(pairs, list):
                return []

            # Validate and normalise each entry
            result: list[dict[str, str]] = []
            for item in pairs:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    result.append(
                        {
                            "question": str(item["question"]),
                            "answer": str(item["answer"]),
                            "difficulty": str(item.get("difficulty", "medium")),
                        }
                    )
            return result

        except Exception as exc:
            print(f"[LiteratureScraper] Haiku parsing failed for figure {figure_num}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _process_identifier(self, ident: str) -> list[ImageRecord]:
        """Full pipeline for a single paper: fetch → extract → QA → records."""
        pdf_bytes = self.fetch_pdf(ident)
        if pdf_bytes is None:
            return []

        figures = self.extract_figures(pdf_bytes, source_id=ident)
        if not figures:
            return []

        records: list[ImageRecord] = []
        safe_ident = re.sub(r"[^\w\-.]", "_", ident)
        source_tag = (
            "literature_arxiv"
            if re.match(r"^\d{4}\.\d{4,5}", ident)
            else "literature_doi"
        )

        for fig in figures:
            # Persist image to disk
            img_path = self.output_dir / f"{safe_ident}_fig{fig.figure_num}.png"
            img_path.write_bytes(fig.image_data)

            if not fig.caption:
                continue

            qa_pairs = self.parse_caption_with_haiku(
                caption=fig.caption,
                figure_num=fig.figure_num,
                image_bytes=fig.image_data if self.include_image_in_haiku_call else None,
            )
            if not qa_pairs:
                continue

            for i, qa in enumerate(qa_pairs):
                records.append(
                    ImageRecord(
                        id=f"{safe_ident}_fig{fig.figure_num}_q{i:02d}",
                        image_path=img_path.resolve(),
                        domain=ImageDomain.OTHER,
                        question=qa["question"],
                        answer=qa["answer"],
                        difficulty=qa.get("difficulty"),  # type: ignore[arg-type]
                        source=source_tag,
                        metadata={
                            "source_id": ident,
                            "figure_num": fig.figure_num,
                            "page_num": fig.page_num,
                            "caption_excerpt": fig.caption[:300],
                        },
                    )
                )
            time.sleep(0.15)  # small courtesy delay between Haiku calls

        return records
